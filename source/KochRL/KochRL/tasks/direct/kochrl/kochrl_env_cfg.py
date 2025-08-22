from .koch import KOCH_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import torch
from isaaclab.sensors import ContactSensorCfg
from .domain_randomization import EventCfg, action_noise_model, observation_noise_model
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

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
    episode_length_s = 12.0
    sample_per_episode = 4
    # sampling
    sampling_origin = [0.025, 0.0, 0.05]
    sampling_radius = 0.20 # 28 cms 
    stiffness_range = [30.0, 1000.0]
    force_range = [0.1, 10.0]  # N
    # - spaces definition
    action_space = 6
    observation_space = 52
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = KOCH_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # domain randomization
    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = action_noise_model
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = observation_noise_model

    # custom parameters/scales
    joints = "joint_[1-6]"
   
    # THE 4 CORE REWARD PARAMETERS FROM SUCCESSFUL REACH ENVIRONMENT
    rew_position_error_weight = -0.2 / 3.0     # L2 position error penalty
    rew_position_tanh_weight = 0.1      # Tanh position reward
    rew_position_tanh_std = 0.3          # Standard deviation for tanh
    rew_orientation_error_weight = 0  # Orientation error penalty
    rew_action_rate_weight = -0.0001     # Action rate penalty
 
    # - reset states/conditions
    shoulder_pan_reset_angles = [-3.140, 3.140]
    shoulder_lift_reset_angles = [-2.094, 0.698]
    elbow_flex_reset_angles = [-1.501, 1.798]
    wrist_flex_reset_angles = [-2.059, 1.815]
    wrist_roll_reset_angles = [-3.140, 3.140]
    gripper_reset_angles = [-1.745, 0.0873]
    total_reset_angles = [shoulder_pan_reset_angles, shoulder_lift_reset_angles, elbow_flex_reset_angles, wrist_flex_reset_angles, wrist_roll_reset_angles, gripper_reset_angles]