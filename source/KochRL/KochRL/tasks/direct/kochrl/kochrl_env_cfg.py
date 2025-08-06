from .koch import KOCH_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import torch

""" 
ACTION SPACE:
[output_q] (6)
"""

"""
OBSERVATION SPACE:
Joint positions: q = [q1, q2, ..., q6] (6)
Joint velocities: q_dot = [q_dot1, q_dot2, ..., q_dot6] (6)

End-effector position: [x, y, z, qx, qy, qz, qw] (7)
End-effector keypoints position: [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] (9)
End-effector linear velocity: [vx, vy, vz] (3)
End-effector angular velocity: [wx, wy, wz] (3)

Target relative Cartesian position: [x_target_err, y_target_err, z_target_err] (3)
Target relative Cartesian keypoints position: [x1_target_err, y1_target_err, z1_target_err], [x2_target_err, y2_target_err, z2_target_err], [x3_target_err, y3_target_err, z3_target_err] (9)

Desired stiffness parameter: k_desired (1)
Previous action: action_prev (6)

Total: 53
"""



@configclass
class KochrlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    sample_per_episode = 4
    # workspace
    workspace_x = [-0.3, 0.3]
    workspace_y = [-0.3, 0.3]
    workspace_z = [0.0, 0.3]
    stiffness_range = [0.0, 100.0]
    # - spaces definition
    action_space = 6
    observation_space = 53
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = KOCH_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    # shoulder_pan = "joint_1"
    # shoulder_lift = "joint_2"
    # elbow_flex = "joint_3"
    # wrist_flex = "joint_4"
    # wrist_roll = "joint_5"
    # gripper = "joint_6"
    joints = "joint_[1-6]"
   
    # - reward scales
    rew_position_err = -0.2
    rew_vel_penalty = -0.0001
    rew_acc_penalty = -0.0001
    rew_out_of_bound_penalty = -10
    # rew_ref_pos_err = 0.1
    # rew_ref_vel_err = 0.075
    # - reset states/conditions
    shoulder_pan_reset_angles = [-3.140, 3.140] #[0, 6.28]  # sample range on reset [rad]
    shoulder_lift_reset_angles = [-2.094, 2.269] #[2.46, 4.71]
    elbow_flex_reset_angles = [-3.140, 3.140] #[1.57, 4.92]
    wrist_flex_reset_angles = [-3.140, 3.140] #[2.79, 6.28]
    wrist_roll_reset_angles = [-3.140, 3.140] #[0, 6.28]
    gripper_reset_angles = [-2.450, 0.032] #[4.6, 6.49]
    total_reset_angles = torch.tensor([shoulder_pan_reset_angles, shoulder_lift_reset_angles, elbow_flex_reset_angles, wrist_flex_reset_angles, wrist_roll_reset_angles, gripper_reset_angles])