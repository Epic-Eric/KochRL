from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .kochrl_env_cfg import KochrlEnvCfg
from .helper import clamp_actions, is_out_of_bound, get_keypoints, sample_target_point, sample_stiffness


class KochrlEnv(DirectRLEnv):
    cfg: KochrlEnvCfg

    def __init__(self, cfg: KochrlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint space
        self._joints_idx, _ = self.robot.find_joints(self.cfg.joints)
        self.num_joints = len(self._joints_idx)
        self.total_reset_angles = torch.tensor(self.cfg.total_reset_angles, device=self.device)
        
        # Initialize tensors with proper shapes
        self.joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Cartesian space
        self.ee_body_idx = self.robot.find_bodies("link_6")[0][0]
        self.ee_body_pos = torch.zeros((self.num_envs, 7), device=self.device)
        self.ee_linear_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angular_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.keypoints = torch.zeros((self.num_envs, 9), device=self.device)
        
        # Task
        self.sampled_target_pos = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_err = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_keypoint_err = torch.zeros((self.num_envs, 9), device=self.device)
        self.k_stiffness = torch.zeros((self.num_envs, 1), device=self.device)

        # Other
        self.prev_action = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        
        # Sample tracking
        self.samples_per_episode = self.cfg.sample_per_episode
        self.steps_per_sample = int(self.max_episode_length / self.samples_per_episode)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Actions are delta joint positions, add to current positions, and clamp to limits
        self.clamped_targets = clamp_actions(self.actions + self.joint_pos[:, self._joints_idx], self.total_reset_angles)
        
        # Set joint position targets
        self.robot.set_joint_position_target(self.clamped_targets, joint_ids=self._joints_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (   
                # Joint space (12)
                self.joint_pos[:, self._joints_idx],
                self.joint_vel[:, self._joints_idx],

                # Cartesian space (22)
                self.ee_body_pos,        # 7
                self.ee_linear_vel,      # 3
                self.ee_angular_vel,     # 3
                self.keypoints,          # 9

                # Task params (13)
                self.target_err,         # 3
                self.target_keypoint_err,# 9
                self.k_stiffness,        # 1

                # Other (6)
                self.prev_action         # 6
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_position_err,
            self.cfg.rew_vel_penalty,
            self.cfg.rew_acc_penalty,
            self.cfg.rew_out_of_bound_penalty,
            self.target_keypoint_err,
            self.joint_pos[:, self._joints_idx],
            self.joint_vel[:, self._joints_idx],
            self.joint_acc[:, self._joints_idx],
            self.total_reset_angles
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Updates - get fresh data from simulation
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_acc = self.robot.data.joint_acc
        
        # Cartesian space
        self.ee_body_pos = self.robot.data.body_pose_w[:, self.ee_body_idx, :]
        self.ee_linear_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, :3]
        self.ee_angular_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, 3:]
        self.keypoints = get_keypoints(self.ee_body_pos)
        
        # Task - compute errors
        self.target_err = self.ee_body_pos[:, :3] - self.sampled_target_pos[:, :3]
        self.target_keypoint_err = self.keypoints - get_keypoints(self.sampled_target_pos)
        
        # Store previous action
        self.prev_action = self.actions.clone()

        # Resample target and stiffness at specified intervals
        current_step = self.episode_length_buf
        resample_mask = (current_step > 0) & ((current_step + 1) % self.steps_per_sample == 0)
        if (torch.any(resample_mask)):
            resample_list = torch.where(resample_mask)[0]
            # sample new targets for all environments
            temp = sample_target_point(self.cfg.workspace_x, self.cfg.workspace_y, self.cfg.workspace_z).to(self.device)
            # first 3 pos xyz
            self.sampled_target_pos[resample_list, :3] = (self.scene.env_origins[resample_list] + temp[:3])
            # last 4 quat xyzw
            self.sampled_target_pos[resample_list, 3:] = temp[3:]
            # stiffness
            self.k_stiffness[resample_list] = sample_stiffness(self.cfg.stiffness_range, len(resample_list)).to(self.device)

        # Check termination conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if joints are out of bounds
        out_of_bounds = is_out_of_bound(self.joint_pos[:, self._joints_idx], self.total_reset_angles)
        
        # Check if end effector is below ground
        ee_below_ground = self.ee_body_pos[:, 2] <= 0.0
        out_of_bounds = out_of_bounds | ee_below_ground
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Sample random joint positions within limits
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        limits = self.total_reset_angles.to(self.device)
        
        # Create proper min/max tensors
        min_limits = limits[:, 0].unsqueeze(0).expand(len(env_ids), -1)
        max_limits = limits[:, 1].unsqueeze(0).expand(len(env_ids), -1)
        
        # Sample uniform random positions
        joint_pos = sample_uniform(
            min_limits,
            max_limits,
            joint_pos.shape,
            self.device
        )
        
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Set root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.joint_acc[env_ids] = 0.0
        
        # Reset task parameters
        temp = sample_target_point(self.cfg.workspace_x, self.cfg.workspace_y, self.cfg.workspace_z).to(self.device)
        for i, env_id in enumerate(env_ids):
            # first 3 pos xyz
            self.sampled_target_pos[env_id, :3] = (self.scene.env_origins[env_id] +  temp[:3])
            # last 4 quat xyzw
            self.sampled_target_pos[env_id, 3:] = temp[3:]
        
        self.k_stiffness[env_ids] = sample_stiffness(self.cfg.stiffness_range, len(env_ids)).to(self.device)
        
        # Reset other states
        self.prev_action[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_pos_err: float,
    rew_scale_vel_penalty: float,
    rew_scale_acc_penalty: float,
    rew_scale_out_of_bound_penalty: float,
    target_keypoint_err: torch.Tensor,
    joints_pos: torch.Tensor,
    joints_vel: torch.Tensor,
    joints_acc: torch.Tensor,
    joints_pos_limit: torch.Tensor
) -> torch.Tensor:
    # Position error 
    rew_pos_err = torch.sum(torch.square(target_keypoint_err), dim=-1) * rew_scale_pos_err
    
    # Velocity penalty
    rew_vel_penalty = torch.sum(torch.square(joints_vel), dim=-1) * rew_scale_vel_penalty
    
    # Acceleration penalty
    rew_acc_penalty = torch.sum(torch.square(joints_acc), dim=-1) * rew_scale_acc_penalty
    
    # Out of bounds penalty
    rew_out_of_bound = is_out_of_bound(joints_pos, joints_pos_limit).float() * rew_scale_out_of_bound_penalty

    total_reward = rew_pos_err + rew_vel_penalty + rew_acc_penalty + rew_out_of_bound
    return total_reward