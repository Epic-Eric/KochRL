#!/usr/bin/env python3
"""
Teleoperation runner that extends lerobot teleop to use a trained RL policy.

Flow:
- Read leader joint angles (deg) via KochLeader
- Read follower joint angles (rad) and compute EE pose [x,y,z,qx,qy,qz,qw]
- Build KochRL-style observation vector
- Run policy inference to get action (delta joint pos for 6 joints)
- Send resultant joint targets to follower arm

Notes:
- Observation layout matches KochrlEnv._get_observations()
- Uses helper.get_keypoints for 3 EE keypoints (9 dims)
"""

import os
import sys
import time
import argparse
import numpy as np
np.set_printoptions(suppress=True)
import torch
# Local hardware control
# Add KochRL source for helper and policy loader
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_DIR = os.path.join(ROOT_DIR, "source", "KochRL")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


# Policy loader used for rsl_rl checkpoints
from load_policy_inference import PolicyLoader  # noqa: E402

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus, TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from custom_code.configure_koch import KochLeader, KochFollower
import subprocess

class SimpleLeader:
    def __init__(self, port: str):
        try:
            subprocess.run(["sudo", "chmod", "666", port], check=True, text=True, capture_output=True)
        except Exception:
            pass
        leader_arm = DynamixelMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )
        self.robot = ManipulatorRobot(
            robot_type="koch",
            leader_arms={"main": leader_arm},
            calibration_dir=".cache/calibration/koch",
        )
        self.robot.connect()
        try:
            self.robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        except Exception:
            pass
    def get_arm_joint_angles(self) -> np.ndarray:
        angles = self.robot.leader_arms["main"].read("Present_Position")
        return np.array(angles)

class SimpleFollower:
    def __init__(self, port: str, torque: bool=True):
        try:
            subprocess.run(["sudo", "chmod", "666", port], check=True, text=True, capture_output=True)
        except Exception:
            pass
        follower_arm = DynamixelMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )
        self.robot = ManipulatorRobot(
            robot_type="koch",
            follower_arms={"main": follower_arm},
            calibration_dir=".cache/calibration/koch",
        )
        self.robot.connect()
        # Position mode and torque
        try:
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
            self.robot.follower_arms["main"].write("Operating_Mode", 3)
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value if torque else TorqueMode.DISABLED.value)
        except Exception:
            pass
    def get_arm_joint_angles(self) -> np.ndarray:
        angles_deg = self.robot.follower_arms["main"].read("Present_Position")
        return np.array(angles_deg) / 360.0 * 2.0 * np.pi


def initialization(follower: SimpleFollower, leader: SimpleLeader):
    try:
        follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        follower.robot.follower_arms["main"].write("Operating_Mode", 3)
        follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
        leader_angles = leader.get_arm_joint_angles()
        follower.robot.follower_arms["main"].write("Goal_Position", leader_angles)
        import time as _t; _t.sleep(2)
    except Exception:
        pass



# Constants: fixed serial ports and action scale
LEADER_PORT = "/dev/ttyACM1"
FOLLOWER_PORT = "/dev/ttyACM0"
ACTION_SCALE = 0.5


def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must be length 4 [qx,qy,qz,qw]")
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)



def quat_rotate_vector_torch(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.dim() == 2:
        vec = vec.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    qx = quat[:, 0:1]
    qy = quat[:, 1:2]
    qz = quat[:, 2:3]
    qw = quat[:, 3:4]
    q_xyz = torch.cat([qx, qy, qz], dim=1).unsqueeze(1)
    cross1 = torch.cross(q_xyz.expand_as(vec), vec, dim=-1)
    cross1_plus_qw_v = cross1 + qw.unsqueeze(-1) * vec
    cross2 = torch.cross(q_xyz.expand_as(vec), cross1_plus_qw_v, dim=-1)
    rotated = vec + 2.0 * cross2
    if squeeze_output:
        rotated = rotated.squeeze(1)
    return rotated


def get_keypoints_torch(ee_body_pos: torch.Tensor) -> torch.Tensor:
    assert ee_body_pos.shape[-1] == 7, "End effector pose must be [x,y,z,qx,qy,qz,qw]"
    batch_size = ee_body_pos.shape[0]
    device = ee_body_pos.device
    pos = ee_body_pos[:, 0:3]
    quat = ee_body_pos[:, 3:7]
    local_offsets = torch.tensor([[0.05, 0.0, 0.0],
                                  [0.0, 0.05, 0.0],
                                  [0.0, 0.0, 0.05]], device=device, dtype=torch.float32)
    world_offsets = quat_rotate_vector_torch(quat, local_offsets.unsqueeze(0).repeat(batch_size, 1, 1))
    keypoints = torch.zeros((batch_size, 9), dtype=torch.float32, device=device)
    keypoints[:, 0:3] = pos + world_offsets[:, 0, :]
    keypoints[:, 3:6] = pos + world_offsets[:, 1, :]
    keypoints[:, 6:9] = pos + world_offsets[:, 2, :]
    return keypoints

def compute_keypoints_np(ee_pose_xyzw: np.ndarray) -> np.ndarray:
    """Compute 3 keypoints (9 dims) in numpy using torch helper for consistency."""
    ee_pose_tensor = torch.tensor(ee_pose_xyzw, dtype=torch.float32).unsqueeze(0)
    kpts = get_keypoints_torch(ee_pose_tensor)  # [1,9]
    return kpts.squeeze(0).cpu().numpy()


def rotmat_to_quat_xyzw(Rm: np.ndarray) -> np.ndarray:
    m00, m01, m02 = Rm[0, 0], Rm[0, 1], Rm[0, 2]
    m10, m11, m12 = Rm[1, 0], Rm[1, 1], Rm[1, 2]
    m20, m21, m22 = Rm[2, 0], Rm[2, 1], Rm[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    return normalize_quaternion_xyzw(q)


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=np.float32)


def build_observation(
    follower_joint_pos_rad: np.ndarray,
    follower_joint_vel_rad: np.ndarray,
    ee_pose_xyzw: np.ndarray,
    ee_lin_vel: np.ndarray,
    ee_ang_vel: np.ndarray,
    target_err_xyz: np.ndarray,
    target_keypoint_err: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Assemble the 52-dim observation expected by the policy.

    Layout from KochrlEnv:
      - Joint positions (6)
      - Joint velocities (6)
      - EE pose xyz+quat (7)
      - EE linear vel (3)
      - EE angular vel (3)
      - Keypoints (9)
      - Target error (3)            -> follower - leader (xyz)
      - Target keypoint error (9)   -> follower - leader keypoints
      - Previous action (6)
    Total = 52
    """
    # Ensure sizes
    q = np.asarray(follower_joint_pos_rad, dtype=np.float32).reshape(6)
    qd = np.asarray(follower_joint_vel_rad, dtype=np.float32).reshape(6)
    ee_pose = np.asarray(ee_pose_xyzw, dtype=np.float32).reshape(7)
    ee_v = np.asarray(ee_lin_vel, dtype=np.float32).reshape(3)
    ee_w = np.asarray(ee_ang_vel, dtype=np.float32).reshape(3)
    t_err = np.asarray(target_err_xyz, dtype=np.float32).reshape(3)
    tk_err = np.asarray(target_keypoint_err, dtype=np.float32).reshape(9)
    prev_u = np.asarray(prev_action, dtype=np.float32).reshape(6)

    # Keypoints from helper for consistency with training
    keypoints = compute_keypoints_np(ee_pose)

    obs_parts = [q, qd, ee_pose, ee_v, ee_w, keypoints, t_err, tk_err, prev_u]
    obs = np.concatenate(obs_parts, axis=0)
    assert obs.shape[0] == 52, f"Observation dim mismatch: {obs.shape[0]}"
    return obs


def clamp_joint_targets_rad(targets: np.ndarray, limits_rad: np.ndarray) -> np.ndarray:
    """Clamp joint targets between joint limits."""
    return np.clip(targets, limits_rad[:, 0], limits_rad[:, 1])


def degrees_to_device_units_deg(deg_vals: np.ndarray) -> np.ndarray:
    """Utility to ensure values sent to Dynamixel Goal_Position are in degrees.
    ManipulatorRobot.write("Goal_Position", ...) expects degrees for these bus configs.
    Input array in degrees.
    """
    return np.asarray(deg_vals, dtype=float)

def real_angle_to_sim_angle_radians(real_angle: np.ndarray) -> np.ndarray:
    real_shoulder_pan, real_shoulder_lift, real_elbow_flex, real_wrist_flex, real_wrist_roll, real_gripper = real_angle
    # shoulder pan
    sim_shoulder_pan = np.pi / 2 - real_shoulder_pan
    # shoulder lift
    sim_shoulder_lift = real_shoulder_lift - np.pi / 2
    # elbow flex
    sim_elbow_flex = np.pi / 2 - real_elbow_flex
    # wrist flex
    sim_wrist_flex = -real_wrist_flex
    # wrist roll
    sim_wrist_roll = real_wrist_roll - np.pi / 2
    # gripper
    sim_gripper = -real_gripper
    return np.array([sim_shoulder_pan, sim_shoulder_lift, sim_elbow_flex, sim_wrist_flex, sim_wrist_roll, sim_gripper])

def sim_angle_to_real_angle_radians(sim_angle: np.ndarray) -> np.ndarray:
    sim_shoulder_pan, sim_shoulder_lift, sim_elbow_flex, sim_wrist_flex, sim_wrist_roll, sim_gripper = sim_angle
    real_shoulder_pan = np.pi / 2 - sim_shoulder_pan
    real_shoulder_lift = sim_shoulder_lift + np.pi / 2
    real_elbow_flex = np.pi / 2 - sim_elbow_flex
    real_wrist_flex = -sim_wrist_flex
    real_wrist_roll = sim_wrist_roll + np.pi / 2
    real_gripper = -sim_gripper
    return np.array([real_shoulder_pan, real_shoulder_lift, real_elbow_flex, real_wrist_flex, real_wrist_roll, real_gripper])


def run_loop(
    follower: KochFollower,
    leader: KochLeader,
    policy: PolicyLoader,
    rate_hz: float,
    action_scale: float,
    joint_limits_rad: np.ndarray,
):
    dt = 1.0 / max(rate_hz, 1e-3)
    prev_action = np.zeros(6, dtype=np.float32)

    # Simple velocity estimates via finite differences
    last_q = follower.get_arm_joint_angles()  # radians
    last_time = time.time()

    # Build DH robot once (same kinematics for leader and follower)
    d1, a2, a3, d5 = 5.5, 10.68, 10.0, 10.5
    A = np.array([0.0, a2, a3, 0.0, 0.0], dtype=np.float32)
    ALPHA = np.array([-np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, 0.0], dtype=np.float32)
    D = np.array([d1, 0.0, 0.0, 0.0, d5], dtype=np.float32)
    OFFSET = np.array([0.0, 0.0, 0.0, np.pi / 2.0, 0.0], dtype=np.float32)
    def map_to_dh_joint_order(q_rad_6: np.ndarray) -> np.ndarray:
        q5 = np.array(q_rad_6[:5], dtype=float)
        q5[4] = np.abs(q5[4])
        q5 = np.round(q5, 2)
        q5 *= np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
        return q5

    def fk_ee_pose_from_q_rad(q_rad_6: np.ndarray) -> np.ndarray:
        q5 = map_to_dh_joint_order(q_rad_6)
        T = np.eye(4, dtype=np.float32)
        for i in range(5):
            Ti = dh_matrix(float(A[i]), float(ALPHA[i]), float(D[i]), float(q5[i] + OFFSET[i]))
            T = T @ Ti
        ee_pos = T[:3, 3] / 100.0 # convert to meters
        ee_quat_xyzw = rotmat_to_quat_xyzw(T[:3, :3])
        return np.hstack((ee_pos, ee_quat_xyzw)).astype(np.float32)

    def fk_ee_pose_from_leader_deg(leader_deg_6: np.ndarray) -> np.ndarray:
        q_rad = np.asarray(leader_deg_6, dtype=float) * np.pi / 180.0
        return fk_ee_pose_from_q_rad(q_rad)

    while True:
        now = time.time()
        # Leader: compute EE pose via FK to define target
        leader_deg = leader.get_arm_joint_angles()
        leader_ee_pose = fk_ee_pose_from_leader_deg(leader_deg)  # [x,y,z,qx,qy,qz,qw]
        # REAL TO SIM
        leader_ee_pose[0:2] = leader_ee_pose[0:2][::-1]

        # Follower state
        q_rad = follower.get_arm_joint_angles()  # radians (6,)

        # Follower EE pose via FK (to avoid prints in follower.get_ee_pose)
        ee_pose = fk_ee_pose_from_q_rad(q_rad)
        # REAL TO SIM
        # swap ee_pose[0] and ee_pose[1]
        ee_pose[0:2] = ee_pose[0:2][::-1]

        # Velocities (finite difference for joints; EE vel zeros for simplicity)
        qd_rad = (q_rad - last_q) / max(now - last_time, 1e-3)
        ee_lin_vel = np.zeros(3, dtype=np.float32)
        ee_ang_vel = np.zeros(3, dtype=np.float32)

        # Keypoints and target errors (follower - leader)
        follower_kpts = compute_keypoints_np(ee_pose)
        leader_kpts = compute_keypoints_np(leader_ee_pose)
        target_err_xyz = ee_pose[:3] - leader_ee_pose[:3]
        target_keypoint_err = follower_kpts - leader_kpts

        # REAL TO SIM CONVERSION
        q_rad = real_angle_to_sim_angle_radians(q_rad) # real to sim radian angle conversion


        # Build observation and run policy
        obs = build_observation(
            follower_joint_pos_rad=q_rad,
            follower_joint_vel_rad=qd_rad,
            ee_pose_xyzw=ee_pose,
            ee_lin_vel=ee_lin_vel,
            ee_ang_vel=ee_ang_vel,
            target_err_xyz=target_err_xyz,
            target_keypoint_err=target_keypoint_err,
            prev_action=prev_action,
        )
####
        sections = {
            "Joint space (12)": [12],
            "Cartesian space (22)": [7, 3, 3, 9],
            "Task params (12)": [3, 9],
            "Other (6)": [6],
        }

        idx = 0
        print("Observations:")
        for section, sizes in sections.items():
            print(f"  {section}:")
            for size in sizes:
                part = obs[idx:idx+size]
                part = [f"{x:.2f}" for x in part]
                print(f"    {part}")
                idx += size
        print("--------------------------------")
        action = policy.get_action(obs).astype(np.float32)  # delta joint positions (rad)
#####
        # Scale and integrate action to target joints in radians
        delta = action_scale * action
        q_target_rad = q_rad + delta
        q_target_rad = clamp_joint_targets_rad(q_target_rad, joint_limits_rad)
        q_target_rad = sim_angle_to_real_angle_radians(q_target_rad)

        # Convert to degrees for readable debug and Dynamixel API
        q_target_deg = q_target_rad / np.pi * 180.0

        # Readable debug: leader cartesian, follower joints/EE, action in rad/deg, target joints in deg
        try:
            follower_q_deg = q_rad / np.pi * 180.0
            action_deg = action * 180.0 / np.pi
            debug_msg = (
                "[DEBUG]\n"
                f"  leader_xyz         : {np.round(leader_ee_pose[:3], 4).tolist()}\n"
                # f"  follower_q_rad     : {np.round(q_rad, 4).tolist()}\n"
                # f"  follower_q_deg     : {np.round(follower_q_deg, 2).tolist()}\n"
                f"  follower_ee_xyz    : {np.round(ee_pose[:3], 4).tolist()}\n"
                # f"  action_rad         : {np.round(action, 4).tolist()}\n"
                # f"  action_deg         : {np.round(action_deg, 2).tolist()}\n"
                # f"  q_target_deg       : {np.round(q_target_deg, 2).tolist()}\n"
            )
            print(debug_msg, end="")
        except Exception:
            pass

        # Send command
        # follower.robot.follower_arms["main"].write("Goal_Position", degrees_to_device_units_deg(q_target_deg))

        # Book-keeping and sleep to maintain rate
        prev_action = action
        last_q = q_rad
        last_time = now
        # best-effort rate control
        sleep_time = dt - (time.time() - now)
        if sleep_time > 0:
            time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser(description="Teleop RL runner for KochRL + lerobot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (e.g., .../model_999.pt or exported/policy.pt)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--rate", type=float, default=50.0, help="Control loop frequency (Hz)")
    parser.add_argument(
        "--init-align", action="store_true", help="Move follower to leader's joint positions at start"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Instantiate hardware interfaces
    follower = SimpleFollower(port=FOLLOWER_PORT, torque=True)
    leader = SimpleLeader(port=LEADER_PORT)

    # Optional initialization: set follower Goal_Position to leader angles (deg)
    if args.init_align:
        initialization(follower, leader)

    # Enable position mode (Operating_Mode=3) with torque enabled
    follower.robot.follower_arms["main"].write("Torque_Enable", 0)
    follower.robot.follower_arms["main"].write("Operating_Mode", 3)
    follower.robot.follower_arms["main"].write("Torque_Enable", 1)

    # Joint limits in radians from env cfg (mirror of training limits)
    # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    joint_limits_rad = np.array([
        [-3.140, 3.140],
        [-2.094, 0.698],
        [-1.501, 1.798],
        [-2.059, 1.815],
        [-3.140, 3.140],
        [-1.745, 0.0873],
    ], dtype=np.float32)

    # Load policy
    policy = PolicyLoader(checkpoint_path=args.checkpoint, device=args.device)

    try:
        run_loop(
            follower=follower,
            leader=leader,
            policy=policy,
            rate_hz=args.rate,
            action_scale=ACTION_SCALE,
            joint_limits_rad=joint_limits_rad,
        )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            follower.robot.follower_arms["main"].write("Torque_Enable", 0)
            follower.robot.disconnect()
            leader.robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
