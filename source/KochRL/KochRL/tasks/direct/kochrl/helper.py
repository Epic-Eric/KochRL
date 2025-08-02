import torch

def clamp_actions(l1:torch.tensor, limits:torch.tensor) -> torch.tensor:
    assert len(l1) == len(limits)
    min_limits = torch.tensor(limits[0][i] for i in range(len(limits)))
    max_limits = torch.tensor(limits[1][i] for i in range(len(limits)))
    return torch.clamp(l1, min_limits, max_limits)

def is_out_of_bound(l1:torch.tensor, limits:torch.tensor) -> torch.tensor:
    assert len(l1) == len(limits)
    min_limits = torch.tensor(limits[0][i] for i in range(len(limits)))
    max_limits = torch.tensor(limits[1][i] for i in range(len(limits)))
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
    
    # Extract quaternion components
    qx, qy, qz, qw = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
    
    # Quaternion rotation formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    q_xyz = torch.cat([qx, qy, qz], dim=1).unsqueeze(1)  # [batch, 1, 3]
    
    cross1 = torch.cross(q_xyz.expand_as(vec), vec, dim=-1)
    cross1_plus_qw_v = cross1 + qw.unsqueeze(-1) * vec
    cross2 = torch.cross(q_xyz.expand_as(vec), cross1_plus_qw_v, dim=-1)
    
    rotated = vec + 2.0 * cross2
    return rotated.squeeze(1) if rotated.shape[1] == 1 else rotated