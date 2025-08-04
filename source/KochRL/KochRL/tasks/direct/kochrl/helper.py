import torch
import numpy as np

def clamp_actions(l1: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    batch_size = l1.shape[0]
    num_joints = l1.shape[1]
    assert num_joints == len(limits), f"Joint count mismatch: {num_joints} vs {len(limits)}"
    
    # create min/max tensors with proper shape for broadcasting
    min_limits = torch.tensor([limits[i][0] for i in range(num_joints)], 
                             device=l1.device, dtype=l1.dtype)
    max_limits = torch.tensor([limits[i][1] for i in range(num_joints)], 
                             device=l1.device, dtype=l1.dtype)
    
    # expand to match batch size
    min_limits = min_limits.unsqueeze(0).expand(batch_size, -1)
    max_limits = max_limits.unsqueeze(0).expand(batch_size, -1)
    
    return torch.clamp(l1, min_limits, max_limits)

def is_out_of_bound(l1: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    batch_size = l1.shape[0]
    num_joints = l1.shape[1]
    assert num_joints == len(limits), f"Joint count mismatch: {num_joints} vs {len(limits)}"
    
    min_limits = torch.tensor([limits[i][0] for i in range(num_joints)], 
                             device=l1.device, dtype=l1.dtype)
    max_limits = torch.tensor([limits[i][1] for i in range(num_joints)], 
                             device=l1.device, dtype=l1.dtype)
    
    min_limits = min_limits.unsqueeze(0).expand(batch_size, -1)
    max_limits = max_limits.unsqueeze(0).expand(batch_size, -1)
    
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

def sample_target_point(workspace_x:list, workspace_y:list, workspace_z:list) -> torch.Tensor:
    """
    Sample a random target point within the specified workspace.
    """
    x = torch.FloatTensor(1).uniform_(workspace_x[0], workspace_x[1])
    y = torch.FloatTensor(1).uniform_(workspace_y[0], workspace_y[1])
    z = torch.FloatTensor(1).uniform_(workspace_z[0], workspace_z[1])
    # Sample a random quaternion for the target point
    # Sample 3 uniform random numbers
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    
    # Marsaglia's method
    sqrt1_u1 = np.sqrt(1 - u1)
    sqrt_u1 = np.sqrt(u1)
    
    qw = sqrt1_u1 * np.sin(2 * np.pi * u2)
    qx = sqrt1_u1 * np.cos(2 * np.pi * u2)
    qy = sqrt_u1 * np.sin(2 * np.pi * u3)
    qz = sqrt_u1 * np.cos(2 * np.pi * u3)
    return torch.tensor([x.item(), y.item(), z.item(), qw, qx, qy, qz], dtype=torch.float32)

def sample_stiffness(stiffness_range:list, num_envs) -> torch.Tensor:
    """
    Sample a list of shape (num_envs, 1) with random stiffness values
    """
    return torch.FloatTensor(num_envs, 1).uniform_(stiffness_range[0], stiffness_range[1])