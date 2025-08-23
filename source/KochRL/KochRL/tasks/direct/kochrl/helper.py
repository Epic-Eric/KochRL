import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import math
import isaaclab.utils.math as m
from isaaclab.utils.math import quat_error_magnitude

def clamp_actions(l1: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    batch_size = l1.shape[0]
    num_joints = l1.shape[1]
    assert num_joints == len(limits), f"Joint count mismatch: {num_joints} vs {len(limits)}"
    
    # create min/max tensors with proper shape for broadcasting
    limits = limits.to(l1.device) 
    min_limits = limits[:, 0].unsqueeze(0).expand(batch_size, -1)
    max_limits = limits[:, 1].unsqueeze(0).expand(batch_size, -1)
    
    return torch.clamp(l1, min_limits, max_limits)

def is_out_of_bound(l1: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    batch_size = l1.shape[0]
    num_joints = l1.shape[1]
    assert num_joints == len(limits), f"Joint count mismatch: {num_joints} vs {len(limits)}"
    limits = limits.to(l1.device) 
    min_limits = limits[:, 0].unsqueeze(0).expand(batch_size, -1)
    max_limits = limits[:, 1].unsqueeze(0).expand(batch_size, -1)
    
    # check if any joint is out of bounds for each environment
    return torch.any((l1 < min_limits) | (l1 > max_limits), dim=1)

def get_keypoints(ee_body_pos: torch.Tensor) -> torch.Tensor:
    """
    Get 3 keypoints from the end effector pose. The keypoints are fixed relative to the 
    end-effector frame and rotate with it to avoid discontinuities.
    """
    assert ee_body_pos.shape[-1] == 7, "End effector position should be of shape [x, y, z, qx, qy, qz, qw]"
    
    batch_size = ee_body_pos.shape[0]
    device = ee_body_pos.device
    
    # Extract position and quaternion
    pos = ee_body_pos[:, 0:3]  # [x, y, z]
    quat = ee_body_pos[:, 3:7]  # [qx, qy, qz, qw]
    
    # Define keypoint offsets in the end-effector's local frame
    # These are simple, unique offsets that won't cause jumps
    local_offsets = torch.tensor([
        [0.05, 0.0, 0.0],   # Along X-axis
        [0.0, 0.05, 0.0],   # Along Y-axis  
        [0.0, 0.0, 0.05],   # Along Z-axis
    ], device=device, dtype=torch.float32)
    
    # Rotate the local offsets by the quaternion to get world frame offsets
    world_offsets = quat_rotate_vector(quat, local_offsets.unsqueeze(0).repeat(batch_size, 1, 1))
    
    # Add rotated offsets to the end-effector position
    keypoints = torch.zeros((batch_size, 9), dtype=torch.float32, device=device)
    keypoints[:, 0:3] = pos + world_offsets[:, 0, :]  # Keypoint 1
    keypoints[:, 3:6] = pos + world_offsets[:, 1, :]  # Keypoint 2  
    keypoints[:, 6:9] = pos + world_offsets[:, 2, :]  # Keypoint 3
    
    return keypoints

def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) by quaternion(s). No discontinuities.
    quat: [batch_size, 4] as [qx, qy, qz, qw]
    vec: [batch_size, num_points, 3] or [batch_size, 3]
    """
    if vec.dim() == 2:
        vec = vec.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Extract quaternion components
    qx, qy, qz, qw = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
    
    # Quaternion rotation formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    q_xyz = torch.cat([qx, qy, qz], dim=1).unsqueeze(1)  # [batch, 1, 3]
    
    cross1 = torch.cross(q_xyz.expand_as(vec), vec, dim=-1)
    cross1_plus_qw_v = cross1 + qw.unsqueeze(-1) * vec
    cross2 = torch.cross(q_xyz.expand_as(vec), cross1_plus_qw_v, dim=-1)
    
    rotated = vec + 2.0 * cross2
    
    if squeeze_output:
        rotated = rotated.squeeze(1)
    
    return rotated

def sample_target_point(sampling_origin:list, sampling_radius:float) -> torch.Tensor:
    """
    Sample a random target point within the specified workspace.
    """
    # the sampling space is a sphere centered at sampling_origin with radius sampling_radius
    # get a point in that space in cartesian coordinate that has z >= 0.0 AND is not in the base
    # Sample from 3D normal distribution
    xyz = torch.randn(3)
    
    # Normalize to unit sphere
    xyz = xyz / torch.norm(xyz)
    
    # Sample radius with correct volume distribution
    u = torch.rand(1)
    r = sampling_radius * (u ** (1/3))
    
    # Scale and translate
    xyz = xyz * r
    x = xyz[0] + sampling_origin[0]
    y = xyz[1] + sampling_origin[1] 
    z = xyz[2] + sampling_origin[2]
    
    # If z < 0, below ground, reflect it to make it positive
    if z < 0.0:
        z = -z
    
    # Check if point is in base (rough box check around origin)
    if x < 0.05 and x > 0.00 and abs(y) < 0.02 and z < 0.07: #inside the base motor or 2nd motor
        x, y, z = 0.1, 0.1, 0.1  # Move to a safe point outside the base

    # Base position
    base_coord = torch.tensor([-0.015, 0, 0.06], dtype=torch.float32)
    target_pos = torch.tensor([x, y, z], dtype=torch.float32)

    # --- Step 1: Define x′ axis (from base to target)
    x_prime = target_pos - base_coord
    x_prime = x_prime / torch.norm(x_prime)

    # --- Step 2: Define z′ by projecting world Z onto plane ⟂ to x′
    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    z_prime = world_z - torch.dot(world_z, x_prime) * x_prime
    if torch.norm(z_prime) < 1e-6:
        # Handle case where x′ is almost vertical → choose arbitrary z′
        z_prime = torch.tensor([0.0, 1.0, 0.0])
    z_prime = z_prime / torch.norm(z_prime)

    # --- Step 3: Get y′ via cross product
    y_prime = torch.cross(z_prime, x_prime)
    y_prime = y_prime / torch.norm(y_prime)

    pitch, roll = torch.rand(2) * 2 * math.pi
    qx, qy, qz, qw = m.quat_mul(m.quat_from_angle_axis(roll, m.quat_apply(m.quat_from_angle_axis(pitch, y_prime), torch.tensor([0.0,0.0,1.0]))), m.quat_from_angle_axis(pitch, y_prime))
    
    return torch.tensor([x, y, z, qx, qy, qz, qw], dtype=torch.float32)

def setup_target_markers() -> VisualizationMarkers:
    """Setup a single visualization marker instance that can handle multiple environments."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="Visuals/TargetMarkers",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0),
            ),
            "target_frame": sim_utils.ConeCfg(
                radius=0.005,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0),
            ),
        }
    )
    return VisualizationMarkers(marker_cfg)

# ============================================================================
# THE 4 CORE REWARD FUNCTIONS FROM SUCCESSFUL REACH ENVIRONMENT
# ============================================================================

def position_command_error(curr_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position and the
    current position. The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    
    Args:
        curr_pos: Current position [batch_size, 3]
        target_pos: Target position [batch_size, 3]
    
    Returns:
        L2 norm of position error [batch_size]
    """
    return torch.norm(curr_pos - target_pos, dim=1)


def position_command_error_tanh(curr_pos: torch.Tensor, target_pos: torch.Tensor, std: float) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position and the
    current position and maps it with a tanh kernel.
    
    Args:
        curr_pos: Current position [batch_size, 3]
        target_pos: Target position [batch_size, 3]
        std: Standard deviation for tanh scaling
    
    Returns:
        Tanh reward [batch_size]
    """
    distance = torch.norm(curr_pos - target_pos, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(curr_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation and the
    current orientation. The orientation error is computed as the shortest
    path between the desired and current orientations.
    
    Args:
        curr_quat: Current quaternion [batch_size, 4] as [qx, qy, qz, qw]
        target_quat: Target quaternion [batch_size, 4] as [qx, qy, qz, qw]
    
    Returns:
        Orientation error [batch_size]
    """
    return quat_error_magnitude(curr_quat, target_quat)


def action_rate_l2(current_action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    """L2 norm of action rate (current action - previous action).
    
    Args:
        current_action: Current action [batch_size, action_dim]
        previous_action: Previous action [batch_size, action_dim]
    
    Returns:
        Action rate penalty [batch_size]
    """
    return torch.norm(current_action - previous_action, dim=1)

def action_acc_l2(current_action: torch.Tensor, previous_action: torch.Tensor, previous_previous_action) -> torch.Tensor:
    """L2 norm of action rate (current action - 2 * previous action + previous_previous_action).
    
    Args:
        current_action: Current action [batch_size, action_dim]
        previous_action: Previous action [batch_size, action_dim]
        previous_previous_action: Previous previous action [batch_size, action_dim]
    
    Returns:
        Action rate penalty [batch_size]
    """
    return torch.norm(current_action - 2 * previous_action + previous_previous_action, dim=1)