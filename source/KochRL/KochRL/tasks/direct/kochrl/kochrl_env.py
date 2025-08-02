# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from helper import clamp_actions, is_out_of_bound


class KochrlEnv(DirectRLEnv):
    cfg: KochrlEnvCfg

    def __init__(self, cfg: KochrlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint space
        self._joints_idx, _ = self.robot.find_joints(self.cfg.joints) # shape (,6)
        self.joint_pos = self.robot.data.joint_pos # shape (4096,6)
        self.joint_vel = self.robot.data.joint_vel # shape (4096,6)
        self.joint_acc = self.robot.data.joint_acc # shape (4096,6)

        # Cartesian space
        self.ee_body_idx = self.robot.find_bodies("link_6")[0][0]
        self.ee_body_pos = self.robot.data.body_pos_w[:, self.ee_body_idx, :] # shape (4096, 7)
        self.ee_linear_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, :3] # shape (4096,3)
        self.ee_angular_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, 3:] # shape (4096,3)
        self.keypoints:torch.tensor #shape (4096, 9)
        # Task
        self.target_err:torch.tensor # shape (4096,3)
        self.target_keypoint_err:torch.tensor # shape (4096,9)
        self.k_stiffness: float # shape (4096,1)

        # Other
        self.prev_action:torch.tensor # shape (4096,6)

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
        self.restrained_actions = clamp_actions(self.actions + self.joint_pos, self.cfg.total_reset_angles)
        self.robot.set_joint_position_target(self.restrained_actions, joint_ids=self._joints_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (   
                # Joint space
                self.joint_pos[:, self._joints_idx[0]:self._joints_idx[len(self._joints_idx)-1]],
                self.joint_vel[:, self._joints_idx[0]:self._joints_idx[len(self._joints_idx)-1]],

                # Cartesian space
                self.ee_body_pos,
                self.ee_linear_vel,
                self.ee_angular_vel,
                self.keypoints,

                # Task params
                self.target_err,
                self.target_keypoint_err,
                self.k_stiffness.unsqueeze(dim=1),

                # Other
                self.prev_action
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
            self.joint_pos[:, self._joints_idx[0]:self._joints_idx[len(self._joints_idx)-1]],
            self.joint_vel[:, self._joints_idx[0]:self._joints_idx[len(self._joints_idx)-1]],
            self.joint_acc[:, self._joints_idx[0]:self._joints_idx[len(self._joints_idx)-1]],
            self.cfg.total_reset_angles
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Updates
        ## Joint space
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_acc = self.robot.data.joint_acc
        ## Cartesian space
        self.ee_body_pos = self.robot.data.body_pos_w[:, self.ee_body_idx, :]
        self.ee_linear_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, :3]
        self.ee_angular_vel = self.robot.data.body_vel_w[:, self.ee_body_idx, 3:]

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

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
):
    rew_pos_err = torch.sum(torch.square(target_keypoint_err).unsqueeze(dim=1), dim=-1) * rew_scale_pos_err
    rew_vel_penalty = torch.sum(torch.square(joints_vel).unsqueeze(dim=1), dim=-1) * rew_scale_vel_penalty
    rew_acc_penalty = torch.sum(torch.square(joints_acc).unsqueeze(dim=1), dim=-1) * rew_scale_acc_penalty
    rew_out_of_bound = is_out_of_bound(joints_pos, joints_pos_limit) * rew_scale_out_of_bound_penalty

    total_reward = rew_pos_err + rew_vel_penalty + rew_acc_penalty + rew_out_of_bound
    return total_reward