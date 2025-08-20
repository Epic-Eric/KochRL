from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .kochrl_env_cfg import KochrlEnvCfg
from .helper import clamp_actions, is_out_of_bound, get_keypoints, sample_target_point, sample_stiffness, setup_target_markers, setup_force_markers, v1_to_v2_quaternion
from .helper import compute_self_collision_forces, sample_external_force,compute_spring_equilibrium_position, has_hit_da_g_spot
from .helper import position_command_error, position_command_error_tanh, orientation_command_error, action_rate_l2

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
        self.sampling_forces = torch.zeros((self.num_envs, 3), device=self.device)
        self.spring_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.has_external_force = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # Other
        self.prev_action = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.applied_torque = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        # self.action_buffer = torch.zeros((1, self.num_envs, self.num_joints), device=self.device)
        # self.action_buffer_idx = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
        # Sample tracking
        self.samples_per_episode = self.cfg.sample_per_episode
        self.steps_per_sample = int(self.max_episode_length / self.samples_per_episode)

        # Visualization markers
        self.target_markers:VisualizationMarkers
        self.force_markers:VisualizationMarkers

        # Self-collision forces
        self.self_collision_forces = torch.zeros((self.num_envs, 1), device=self.device)

        # Logging for the 4 reach environment rewards
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "rew_position_error",
                "rew_position_tanh",
                "rew_orientation_error", 
                "rew_action_rate",
                "total_reward",
            ]
        }

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
        # Create contact sensors - one sensor per body, each monitoring against all other bodies
        self.contact_sensors = {}
        link_names = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
        
        # Create all filter paths (all robot bodies)
        all_robot_bodies = [f"/World/envs/env_.*/Robot/{name}" for name in link_names]
        
        for i, link_name in enumerate(link_names):
            # Each sensor monitors ONE body against ALL robot bodies (including itself)
            contact_sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",  # ONE sensor body
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=all_robot_bodies,  # Against ALL robot bodies (1-to-many)
            )
            
            sensor = ContactSensor(contact_sensor_cfg)
            self.contact_sensors[link_name] = sensor
            self.scene.sensors[f"contact_sensor_{i}"] = sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add markers
        self.target_markers = setup_target_markers()
        self.force_markers = setup_force_markers()
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # self.actions = self.action_buffer[0]
        # self.action_buffer = torch.cat((self.action_buffer[1:], actions.unsqueeze(0)), dim=0)
        self.prev_action = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Actions are delta joint positions, add to current positions, and clamp to limits
        self.clamped_targets = clamp_actions(self.actions * 0.5 + self.joint_pos[:, self._joints_idx], self.total_reset_angles)
        
        # Set joint position targets
        self.robot.set_joint_position_target(self.clamped_targets, joint_ids=self._joints_idx)
        print(self.actions)

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

                # Other (12)
                self.prev_action,         # 6
                self.applied_torque      # 6
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # THE 4 CORE REWARD FUNCTIONS FROM SUCCESSFUL REACH ENVIRONMENT
        
        # 1. Position error penalty (L2 norm)
        rew_position_error = position_command_error(
            self.ee_body_pos[:, :3], 
            self.sampled_target_pos[:, :3]
        ) * self.cfg.rew_position_error_weight
        
        # 2. Position tracking reward (tanh)
        rew_position_tanh = position_command_error_tanh(
            self.ee_body_pos[:, :3], 
            self.sampled_target_pos[:, :3], 
            self.cfg.rew_position_tanh_std
        ) * self.cfg.rew_position_tanh_weight
        
        # 3. Orientation error penalty
        rew_orientation_error = orientation_command_error(
            self.ee_body_pos[:, 3:7], 
            self.sampled_target_pos[:, 3:7]
        ) * self.cfg.rew_orientation_error_weight
        
        # 4. Action rate penalty
        rew_action_rate = action_rate_l2(
            self.actions, 
            self.prev_action
        ) * self.cfg.rew_action_rate_weight
        
        # Total reward
        total_reward = rew_position_error + rew_position_tanh + rew_orientation_error + rew_action_rate
        
        # Logging
        self._episode_sums["rew_position_error"] += rew_position_error
        self._episode_sums["rew_position_tanh"] += rew_position_tanh
        self._episode_sums["rew_orientation_error"] += rew_orientation_error
        self._episode_sums["rew_action_rate"] += rew_action_rate
        self._episode_sums["total_reward"] += total_reward
        
        return total_reward * self.step_dt

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
        self.applied_torque = self.robot.data.applied_torque

        # Get self-collision forces
        self.self_collision_forces = compute_self_collision_forces(self.contact_sensors)

        # Resample target and stiffness at specified intervals
        current_step = self.episode_length_buf
        resample_mask = (current_step > 0) & ((current_step + 1) % self.steps_per_sample == 0)
        if (torch.any(resample_mask)):
            resample_list = torch.where(resample_mask)[0]
            # sample new targets for all environments
            temp = sample_target_point(self.cfg.sampling_origin, self.cfg.sampling_radius).to(self.device)
            # print(f"Target point sampled: {temp}")
            # first 3 pos xyz
            self.sampled_target_pos[resample_list, :3] = (self.scene.env_origins[resample_list] + temp[:3])
            # last 4 quat xyzw
            self.sampled_target_pos[resample_list, 3:] = temp[3:]
            # stiffness
            self.k_stiffness[resample_list] = sample_stiffness(self.cfg.stiffness_range, len(resample_list)).to(self.device)
            # update target markers
            self.target_markers.visualize(self.sampled_target_pos[:, :3].repeat_interleave(2, dim=0), self.sampled_target_pos[:, 3:].repeat_interleave(2, dim=0), marker_indices=torch.tensor([0, 1] * self.sampled_target_pos.shape[0], device=self.device, dtype=torch.int32))
            # force sampling - only for half the environments
            force_mask = torch.rand(len(resample_list), device=self.device) < 0
            has_force_envs = resample_list[force_mask]
            no_force_envs = resample_list[~force_mask]
            self.has_external_force[resample_list] = force_mask

            # Sample forces only for selected environments
            if len(has_force_envs) > 0:
                print(f"[INFO]: Sampling forces for {len(has_force_envs)} environments")
                self.sampling_forces[has_force_envs] = sample_external_force(
                    self.cfg.force_range,
                    len(has_force_envs),
                    self.device
                )
                self.robot.set_external_force_and_torque(
                    forces=self.sampling_forces[has_force_envs].unsqueeze(1), 
                    torques=torch.zeros((len(has_force_envs), 3), device=self.device).unsqueeze(1), 
                    body_ids=[self.ee_body_idx], 
                    env_ids=has_force_envs
                )

            # Set force to zero for environments without external force
            if len(no_force_envs) > 0:
                self.sampling_forces[no_force_envs] = 0.0

            # Compute spring target position
            self.spring_target_pos = torch.where(
                self.has_external_force.unsqueeze(-1),
                compute_spring_equilibrium_position(
                    self.sampled_target_pos, self.sampling_forces, self.k_stiffness,
                    self.cfg.sampling_origin, self.cfg.sampling_radius
                ),
                self.sampled_target_pos[:, :3]  # Use target position when no force
            )

        self.force_markers.visualize(torch.stack([self.ee_body_pos[:, :3], self.sampled_target_pos[:, :3], self.spring_target_pos], dim=1).view(-1, 3), v1_to_v2_quaternion(torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.ee_body_pos.shape[0], 1), self.sampling_forces).repeat_interleave(3, dim=0), marker_indices=torch.tensor([0, 0, 1] * self.sampled_target_pos.shape[0], device=self.device, dtype=torch.int32))
        
        # Check termination conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if joints are out of bounds
        out_of_bounds = is_out_of_bound(self.joint_pos[:, self._joints_idx], self.total_reset_angles)
        
        # Check if end effector is below ground
        ee_below_ground = self.ee_body_pos[:, 2] <= 0.0
        out_of_bounds = out_of_bounds | ee_below_ground

        # print out average norm self.keypoints error
        avg_keypoint_err = torch.mean(torch.norm(self.target_keypoint_err, dim=-1))
        # print(f"[INFO]: Average keypoint error: {avg_keypoint_err.item():.4f}")
        # print(f"[INFO]: Average target error: {torch.mean(torch.norm(self.target_err, dim=-1)).item():.4f}")
        
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Sample random joint positions and velocityies within limits
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Set root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.joint_acc[env_ids] = 0.0
        self.applied_torque[env_ids] = 0.0
        
        # Reset task parameters
        temp = sample_target_point(self.cfg.sampling_origin, self.cfg.sampling_radius).to(self.device)
        # print(f"Target point sampled: {temp}")
        for i, env_id in enumerate(env_ids):
            # first 3 pos xyz
            self.sampled_target_pos[env_id, :3] = (self.scene.env_origins[env_id] +  temp[:3])
            # last 4 quat xyzw
            self.sampled_target_pos[env_id, 3:] = temp[3:]
        
        self.k_stiffness[env_ids] = sample_stiffness(self.cfg.stiffness_range, len(env_ids)).to(self.device)
        # update target markers
        self.target_markers.visualize(self.sampled_target_pos[:, :3].repeat_interleave(2, dim=0), self.sampled_target_pos[:, 3:].repeat_interleave(2, dim=0), marker_indices=torch.tensor([0, 1] * self.sampled_target_pos.shape[0], device=self.device, dtype=torch.int32))
        # force sampling - only for half the environments
        force_mask = torch.rand(len(env_ids), device=self.device) < 0
        has_force_envs = env_ids[force_mask]
        no_force_envs = env_ids[~force_mask]
        self.has_external_force[env_ids] = force_mask

        # Sample forces only for selected environments
        if len(has_force_envs) > 0:
            print(f"[INFO]: Sampling forces for {len(has_force_envs)} environments")
            self.sampling_forces[has_force_envs] = sample_external_force(
                self.cfg.force_range,
                len(has_force_envs),
                self.device
            )
            self.robot.set_external_force_and_torque(
                forces=self.sampling_forces[has_force_envs].unsqueeze(1), 
                torques=torch.zeros((len(has_force_envs), 3), device=self.device).unsqueeze(1), 
                body_ids=[self.ee_body_idx], 
                env_ids=has_force_envs
            )

        # Set force to zero for environments without external force
        if len(no_force_envs) > 0:
            self.sampling_forces[no_force_envs] = 0.0

        # Compute spring target position
        self.spring_target_pos = torch.where(
            self.has_external_force.unsqueeze(-1),
            compute_spring_equilibrium_position(
                self.sampled_target_pos, self.sampling_forces, self.k_stiffness,
                self.cfg.sampling_origin, self.cfg.sampling_radius
            ),
            self.sampled_target_pos[:, :3]  # Use target position when no force
        )
        print(f"[INFO]: Spring target position: {self.spring_target_pos - self.scene.env_origins}")

        self.force_markers.visualize(torch.stack([self.ee_body_pos[:, :3], self.sampled_target_pos[:, :3], self.spring_target_pos], dim=1).view(-1, 3), v1_to_v2_quaternion(torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.ee_body_pos.shape[0], 1), self.sampling_forces).repeat_interleave(3, dim=0), marker_indices=torch.tensor([0, 0, 1] * self.sampled_target_pos.shape[0], device=self.device, dtype=torch.int32))
        
        # Reset other states
        self.prev_action[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        # self.action_buffer_idx[env_ids] = torch.randint(1, 5, (len(env_ids),), device=self.device)
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Average keypoint error"] = torch.mean(torch.norm(self.target_keypoint_err, dim=1), dim=0).item()
        extras["Average target error"] = torch.mean(torch.norm(self.target_err, dim=1), dim=0).item()
        extras["Average spring target error"] = torch.mean(torch.norm(self.spring_target_pos - self.ee_body_pos[:, :3], dim=1), dim=0).item()
        self.extras["log"].update(extras)


