import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import math

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

def quaternion_mult(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    assert q1.shape == q2.shape, "Quaternions must have the same shape"
    
    r1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    r2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    return torch.tensor(
        [r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2,
        r1 * x2 + r2 * x1 + y1 * z2 - z1 * y2,
        r1 * y2 + r2 * y1 + z1 * x2 - x1 * z2,
        r1 * z2 + r2 * z1 + x1 * y2 - y1 * x2]
    )

def sample_target_point(sampling_origin:list, sampling_radius:float) -> torch.Tensor:
    """
    Sample a random target point within the specified workspace.
    """
    # the sampling space is a sphere centered at sampling_origin with radius sampling_radius
    # get a point in that space in cartesian coordinate that has z >= 0.0 AND is not in the base
    x = (torch.rand(1) - 0.5) * 2 * sampling_radius + sampling_origin[0]
    y = (torch.rand(1) - 0.5) * 2 * sampling_radius + sampling_origin[1] 
    z = (torch.rand(1) - 0.5) * 2 * sampling_radius + sampling_origin[2]
    
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

    # --- Step 4: Rotate the base vector (vertical) to be in line with x prime
    base_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # Base vector in world frame
    axis = torch.cross(base_vector, x_prime)  # Axis of rotation
    axis = axis / torch.norm(axis)  # Normalize the axis
    angle = torch.acos(torch.dot(base_vector, x_prime) / torch.norm(base_vector) / torch.norm(x_prime))  # Angle to rotate
    x0, y0, z0 = torch.sin(angle/2) * axis  # Half-angle sine for quaternion
    r0 = torch.cos(angle/2)  # Half-angle cosine for quaternion

    # # --- Step 5: Sample random rotations theta, do operation (r,x,y,z) and make them into quaternions (qx, qy, qz, qw)
    # angle1, angle2 = torch.rand(2) * 2 * math.pi  # Random angles for rotation, angle1 is around y', angle2 is around x'
    # x1, y1, z1 = torch.sin(angle1/2) * y_prime
    # x2, y2, z2 = torch.sin(angle2/2) * x_prime
    # r1, r2 = torch.cos(angle1/2), torch.cos(angle2/2)
    # product = quaternion_mult(torch.tensor([r0, x0, y0, z0]), torch.tensor([r1, x1, y1, z1]))
    # product = product / torch.norm(product)  # Normalize to ensure it's a unit quaternion
    # product = quaternion_mult(product, torch.tensor([r2, x2, y2, z2]))
    # product = product / torch.norm(product)  # Normalize again to ensure it's a unit quaternion
    # qw, qx, qy, qz = product

    #quat = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    quat = torch.tensor([x0, y0, z0,r0], dtype=torch.float32)
    quat = quat / torch.norm(quat)  # Ensure unit quaternion
    
    return torch.tensor([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=torch.float32)

def sample_stiffness(stiffness_range:list, num_envs) -> torch.Tensor:
    """
    Sample a list of shape (num_envs, 1) with random stiffness values
    """
    return torch.FloatTensor(num_envs, 1).uniform_(stiffness_range[0], stiffness_range[1])

def detect_self_collision(robot_articulation) -> torch.Tensor:
    """Detect self-collision by checking contact sensor data."""
    contact_data = robot_articulation.data.contact_force_matrix_w
    contact_threshold = 0.1
    contact_magnitudes = torch.norm(contact_data, dim=-1)
    return torch.any(contact_magnitudes > contact_threshold, dim=-1)

def setup_target_markers() -> VisualizationMarkers:
    """Setup a single visualization marker instance that can handle multiple environments."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="Visuals/TargetMarkers",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0),
            ),
            "target_frame": sim_utils.CylinderCfg(
                radius=0.005,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0),
            ),
        }
    )
    return VisualizationMarkers(marker_cfg)