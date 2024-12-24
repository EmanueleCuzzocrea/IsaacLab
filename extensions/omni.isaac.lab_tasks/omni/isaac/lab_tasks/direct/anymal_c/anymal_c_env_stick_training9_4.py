from __future__ import annotations
import gymnasium as gym
import torch
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
import math
import random
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.math import subtract_frame_transforms
import numpy as np
from scipy.signal import butter, filtfilt
# Pre-defined configs
from omni.isaac.lab_assets.anymal import ANYMAL_STICK_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )


@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ANYMAL_STICK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True, track_pose=True,
    )

    cuboid_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=1000,
                #restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 0.6)),
    )


    # reward scales
    lin_vel_reward_scale_x = 1.0*3
    lin_vel_reward_scale_y = 1.0*3
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05*3
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    joint_vel_reward_scale = 0.0 #-2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.0
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0*2

    track_yaw = 1.0
    force_variance = 0.0 #-0.000001
    force_acceleration = 0.0 #-0.001
    force_min_max = 0.0 #-0.001
    track_force = -0.00001 #-0.00001
    track_force2 = 2.0
    force_jerk = 0.0 #-0.00000001
    force_snap = 0.0
    joint_deviation = -1.0



@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward machines parameters
        self._P = torch.zeros(self.num_envs, 4, device=self.device)
        self._state = torch.zeros(self.num_envs, 1, device=self.device)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._sequenza_target_1 = torch.tensor([1, 0, 0, 1], device=self.device)
        self._sequenza_target_2 = torch.tensor([1, 1, 1, 1], device=self.device)
        self._sequenza_target_3 = torch.tensor([0, 1, 1, 0], device=self.device)
        self._sequenza_target_4 = torch.tensor([1, 1, 1, 1], device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_x_exp",
                "track_lin_vel_y_exp",
                "track_yaw",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "dof_vel_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "force_tracking",
                "force_tracking2",
                "joint_deviation",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._interaction_ids, _ = self._contact_sensor.find_bodies("interaction")

        self.yaw = torch.zeros(self.num_envs, 1, device=self.device)
        
        self._forces_reference = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces_buffer = torch.zeros(self.num_envs, 300, device=self.device)

        self._integrators = torch.zeros(self.num_envs, 1, device=self.device)

        self._level = torch.zeros(self.num_envs, 1, device=self.device)
        self.count = 0.0
        self.count_int = 0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Cuboid
        self._cuboid = RigidObject(self.cfg.cuboid_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)        

    def _get_observations(self) -> dict:
        self._integrators[:, 0] += 0.0003*(self._forces_reference[:, 0] - self._forces[:, 0])
        self._commands[:, 0] = self._integrators[:, 0]

        mask_force__ = self._forces.squeeze(dim=1) > 0.0
        self._commands[mask_force__, 0] *= 0.0


        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self.yaw,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                    self._forces,
                    self._forces_reference,
                    self._P,
                    self._state,
                    self._phase,
                    #self._frequency,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:        
        # yaw tracking
        root_quat_ = self._robot.data.root_quat_w
        w = root_quat_[:, 0]
        x = root_quat_[:, 1]
        y = root_quat_[:, 2]
        z = root_quat_[:, 3]
        
        yaw1 = 2 * (w * z + x * y)
        yaw2 = 1 - 2 * (y * y + z * z)
        self.yaw[:, 0] = torch.atan2(yaw1, yaw2)

        yaw_error = torch.square(self._commands[:, 2] - self.yaw[:, 0])
        yaw_error_mapped = torch.exp(-yaw_error / 0.25)

        # linear velocity tracking_x
        lin_vel_error_x = torch.square(self._commands[:, 0] - self._robot.data.root_lin_vel_b[:, 0])
        lin_vel_error_mapped_x = torch.exp(-lin_vel_error_x / 0.25)
        # linear velocity tracking_y
        lin_vel_error_y = torch.square(self._commands[:, 1] - self._robot.data.root_lin_vel_b[:, 1])
        lin_vel_error_mapped_y = torch.exp(-lin_vel_error_y / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # joint velocity
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # reward machine
        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        mask_contact = current_contact_time > 0.0
        mask_air = current_air_time > 0.0
        self._P[:, :][mask_contact] = 1
        self._P[:, :][mask_air] = 0
        
        self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        mask_phase = (self._phase[:, 0] < 16)
        self._phase[:, 0][mask_phase] += 1

        maschera1 = (self._P == self._sequenza_target_1).all(dim=1) & (self._phase > 0).all(dim=1) & (self._state == 0).all(dim=1)
        self._state[:, 0][maschera1] = 1
        self._extra_reward[maschera1] = 2
        self._phase[:, 0][maschera1] = 0
        maschera2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._phase > 5).all(dim=1) & (self._state == 1).all(dim=1)
        self._state[:, 0][maschera2] = 2
        self._extra_reward[maschera2] = 2
        self._phase[:, 0][maschera2] = 0
        maschera3 = (self._P == self._sequenza_target_3).all(dim=1) & (self._phase > 0).all(dim=1) & (self._state == 2).all(dim=1)
        self._state[:, 0][maschera3] = 3
        self._extra_reward[maschera3] = 2
        self._phase[:, 0][maschera3] = 0
        maschera4 = (self._P == self._sequenza_target_4).all(dim=1) & (self._phase > 5).all(dim=1) & (self._state == 3).all(dim=1)
        self._state[:, 0][maschera4] = 0
        self._extra_reward[maschera4] = 2
        self._phase[:, 0][maschera4] = 0
        self._extra_reward = self._extra_reward.squeeze()

        
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # interaction force
        interaction_force = self._contact_sensor.data.net_forces_w[:, self._interaction_ids].squeeze(dim=1)
        z_component = torch.abs(interaction_force[:, 0])
        z_component *= torch.cos(self.yaw[:, 0])
        self._forces[:,0] = z_component

        # interaction force buffer
        self._forces_buffer[:, :-1] = self._forces_buffer[:, 1:].clone()
        self._forces_buffer[:, -1] = self._forces[:,0].squeeze()

        # force tracking
        force_error = torch.square(self._forces_reference[:, 0] - self._forces[:, 0])
        force_error_mapped = torch.exp(-force_error / 0.25)

        # joint deviation
        deviation = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_deviation = torch.sum(torch.abs(deviation), dim=1)

        rewards = {
            "track_lin_vel_x_exp": lin_vel_error_mapped_x * self.cfg.lin_vel_reward_scale_x * self.step_dt,
            "track_lin_vel_y_exp": lin_vel_error_mapped_y * self.cfg.lin_vel_reward_scale_y * self.step_dt,
            "track_yaw": yaw_error_mapped * self.cfg.track_yaw * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "dof_vel_l2": joint_vel * self.cfg.joint_vel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "force_tracking": force_error * self.cfg.track_force * self.step_dt,
            "force_tracking2": force_error_mapped * self.cfg.track_force2 * self.step_dt,
            "joint_deviation": joint_deviation * self.cfg.joint_deviation * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        mask_extra = self._extra_reward > 0
        reward[mask_extra] *= 2.0

        self.count += 0.001

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(0.0, 0.0)
    
        self._commands[env_ids, 2] = 0.0
        x_ = random.uniform(0.0, 0.3)
        y_ = random.uniform(-0.2, 0.2)
        self._commands[env_ids,0] = 0.0
        self._commands[env_ids,1] = 0.0

        # Sample new force commands
        self._integrators[env_ids] = 0.0
        
        # Reward machines
        self._P[env_ids, :] = 0
        self._state[env_ids, 0] = 0
        self._phase[env_ids, 0] = 0
        self._frequency[env_ids, 0] = random.randint(2, 4)
        self._frequency[env_ids, 0] = random.randint(2, 4)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset cube state
        cube_used = torch.tensor([1.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        cube_pose = self._robot.data.default_root_state[env_ids]
        cube_pose[:, :3] += self._terrain.env_origins[env_ids]
        self._cuboid.write_root_pose_to_sim(cube_pose[:, :7] + cube_used, env_ids)

        # Curriculum learning
        mask_level_up = (self._forces_buffer.min(dim=1).values > (self._forces_reference[:, 0] - 5.0)) & (self._forces_buffer.max(dim=1).values < (self._forces_reference[:, 0] + 5.0))
        selected_mask_up = mask_level_up[env_ids]
        self._level[env_ids[selected_mask_up], 0] += 1

        mask_level_down = (self._forces_buffer.min(dim=1).values <= (self._forces_reference[:, 0] - 5.0)) | (self._forces_buffer.max(dim=1).values >= (self._forces_reference[:, 0] + 5.0))
        selected_mask_down = mask_level_down[env_ids]
        self._level[env_ids[selected_mask_down], 0] -= 1
        
        if (self.count >=  0.0 and self.count < 3.0):
            self._level[:, 0].clamp_(min=0, max=1)
        if (self.count >= 3.0 and self.count < 6.0):
            self._level[:, 0].clamp_(min=0, max=2)
        if (self.count >= 6.0 and self.count < 9.0):
            self._level[:, 0].clamp_(min=0, max=3)
        if (self.count >= 9.0 and self.count < 12.0):
            self._level[:, 0].clamp_(min=0, max=4)
        if (self.count >= 12.0 and self.count < 15.0):
            self._level[:, 0].clamp_(min=0, max=5)
        if (self.count >= 15.0 and self.count < 18.0):
            self._level[:, 0].clamp_(min=0, max=6)
        if (self.count >= 18.0 and self.count < 21.0):
            self._level[:, 0].clamp_(min=0, max=7)
        if (self.count >= 21.0 and self.count < 24.0):
            self._level[:, 0].clamp_(min=0, max=8)
        if (self.count >= 24.0 and self.count < 30.0):
            self._level[:, 0].clamp_(min=0, max=9)
        if (self.count >= 30.0 and self.count < 33.0):
            self._level[:, 0].clamp_(min=0, max=10)
        if (self.count >= 33.0 and self.count < 36.0):
            self._level[:, 0].clamp_(min=0, max=11)
        if (self.count >= 36.0 and self.count < 39.0):
            self._level[:, 0].clamp_(min=0, max=12)
        if (self.count >= 39.0 and self.count < 42.0):
            self._level[:, 0].clamp_(min=0, max=13)
        if (self.count >= 42.0 and self.count < 45.0):
            self._level[:, 0].clamp_(min=0, max=14)
        if (self.count >= 45.0 and self.count < 48.0):
            self._level[:, 0].clamp_(min=0, max=15)
        if (self.count >= 48.0 and self.count < 51.0):
            self._level[:, 0].clamp_(min=0, max=16)
        if (self.count >= 51.0 and self.count < 54.0):
            self._level[:, 0].clamp_(min=0, max=17)
        if (self.count >= 54.0 and self.count < 57.0):
            self._level[:, 0].clamp_(min=0, max=18)

        if (self.count >= 57.0 and self.count < 60.0):
            self._level[:, 0].clamp_(min=0, max=19)
        if (self.count >= 60.0 and self.count < 65.0):
            self._level[:, 0].clamp_(min=0, max=20)
        if (self.count >= 65.0 and self.count < 70.0):
            self._level[:, 0].clamp_(min=0, max=21)
        if (self.count >= 70.0 and self.count < 75.0):
            self._level[:, 0].clamp_(min=0, max=22)
        if (self.count >= 75.0 and self.count < 80.0):
            self._level[:, 0].clamp_(min=0, max=23)
        if (self.count >= 80.0 and self.count < 85.0):
            self._level[:, 0].clamp_(min=0, max=24)
        if (self.count >= 85.0 and self.count < 90.0):
            self._level[:, 0].clamp_(min=0, max=25)
        if (self.count >= 90.0 and self.count < 95.0):
            self._level[:, 0].clamp_(min=0, max=26)
        if (self.count >= 95.0 and self.count < 100.0):
            self._level[:, 0].clamp_(min=0, max=27)
        if (self.count >= 100.0 and self.count < 105.0):
            self._level[:, 0].clamp_(min=0, max=28)
        if (self.count >= 105.0 and self.count < 110.0):
            self._level[:, 0].clamp_(min=0, max=29)
        if (self.count >= 110.0 and self.count < 115.0):
            self._level[:, 0].clamp_(min=0, max=30)
        if (self.count >= 115.0 and self.count < 120.0):
            self._level[:, 0].clamp_(min=0, max=31)
        if (self.count >= 120.0 and self.count < 125.0):
            self._level[:, 0].clamp_(min=0, max=32)
        if (self.count >= 125.0 and self.count < 130.0):
            self._level[:, 0].clamp_(min=0, max=33)
        if (self.count >= 130.0 and self.count < 135.0):
            self._level[:, 0].clamp_(min=0, max=34)
        if (self.count >= 135.0 and self.count < 140.0):
            self._level[:, 0].clamp_(min=0, max=35)
        if (self.count >= 140.0 and self.count < 145.0):
            self._level[:, 0].clamp_(min=0, max=36)
        if (self.count >= 145.0 and self.count < 150.0):
            self._level[:, 0].clamp_(min=0, max=37)
        if (self.count >= 150.0 and self.count < 155.0):
            self._level[:, 0].clamp_(min=0, max=38)

        livello0 = env_ids[(self._level[env_ids, 0] == 0)]
        livello1 = env_ids[(self._level[env_ids, 0] == 1)]
        livello2 = env_ids[(self._level[env_ids, 0] == 2)]
        livello3 = env_ids[(self._level[env_ids, 0] == 3)]
        livello4 = env_ids[(self._level[env_ids, 0] == 4)]
        livello5 = env_ids[(self._level[env_ids, 0] == 5)]
        livello6 = env_ids[(self._level[env_ids, 0] == 6)]
        livello7 = env_ids[(self._level[env_ids, 0] == 7)]
        livello8 = env_ids[(self._level[env_ids, 0] == 8)]
        livello9 = env_ids[(self._level[env_ids, 0] == 9)]
        livello10 = env_ids[(self._level[env_ids, 0] == 10)]
        livello11 = env_ids[(self._level[env_ids, 0] == 11)]
        livello12 = env_ids[(self._level[env_ids, 0] == 12)]
        livello13 = env_ids[(self._level[env_ids, 0] == 13)]
        livello14 = env_ids[(self._level[env_ids, 0] == 14)]
        livello15 = env_ids[(self._level[env_ids, 0] == 15)]
        livello16 = env_ids[(self._level[env_ids, 0] == 16)]
        livello17 = env_ids[(self._level[env_ids, 0] == 17)]
        livello18 = env_ids[(self._level[env_ids, 0] == 18)]
        livello19 = env_ids[(self._level[env_ids, 0] == 19)]
        livello20 = env_ids[(self._level[env_ids, 0] == 20)]
        livello21 = env_ids[(self._level[env_ids, 0] == 21)]
        livello22 = env_ids[(self._level[env_ids, 0] == 22)]
        livello23 = env_ids[(self._level[env_ids, 0] == 23)]
        livello24 = env_ids[(self._level[env_ids, 0] == 24)]
        livello25 = env_ids[(self._level[env_ids, 0] == 25)]
        livello26 = env_ids[(self._level[env_ids, 0] == 26)]
        livello27 = env_ids[(self._level[env_ids, 0] == 27)]
        livello28 = env_ids[(self._level[env_ids, 0] == 28)]
        livello29 = env_ids[(self._level[env_ids, 0] == 29)]
        livello30 = env_ids[(self._level[env_ids, 0] == 30)]
        livello31 = env_ids[(self._level[env_ids, 0] == 31)]
        livello32 = env_ids[(self._level[env_ids, 0] == 32)]
        livello33 = env_ids[(self._level[env_ids, 0] == 33)]
        livello34 = env_ids[(self._level[env_ids, 0] == 34)]
        livello35 = env_ids[(self._level[env_ids, 0] == 35)]
        livello36 = env_ids[(self._level[env_ids, 0] == 36)]
        livello37 = env_ids[(self._level[env_ids, 0] == 37)]
        livello38 = env_ids[(self._level[env_ids, 0] == 38)]
        livello39 = env_ids[(self._level[env_ids, 0] == 39)]
        livello40 = env_ids[(self._level[env_ids, 0] == 40)]
        livello41 = env_ids[(self._level[env_ids, 0] == 41)]
        livello42 = env_ids[(self._level[env_ids, 0] == 42)]
        livello43 = env_ids[(self._level[env_ids, 0] == 43)]
        livello44 = env_ids[(self._level[env_ids, 0] == 44)]
        livello45 = env_ids[(self._level[env_ids, 0] == 45)]
        livello46 = env_ids[(self._level[env_ids, 0] == 46)]
        livello47 = env_ids[(self._level[env_ids, 0] == 47)]
        livello48 = env_ids[(self._level[env_ids, 0] == 48)]

        self._forces_reference[env_ids] = torch.zeros_like(self._forces_reference[env_ids]).uniform_(20.0, 20.0)
        self._forces_reference[livello0] = torch.zeros_like(self._forces_reference[livello0]).uniform_(10.0, 11.0)
        self._forces_reference[livello1] = torch.zeros_like(self._forces_reference[livello1]).uniform_(11.0, 12.0)
        self._forces_reference[livello2] = torch.zeros_like(self._forces_reference[livello2]).uniform_(12.0, 13.0)
        self._forces_reference[livello3] = torch.zeros_like(self._forces_reference[livello3]).uniform_(14.0, 15.0)
        self._forces_reference[livello4] = torch.zeros_like(self._forces_reference[livello4]).uniform_(15.0, 16.0)
        self._forces_reference[livello5] = torch.zeros_like(self._forces_reference[livello5]).uniform_(16.0, 17.0)
        self._forces_reference[livello6] = torch.zeros_like(self._forces_reference[livello6]).uniform_(17.0, 18.0)
        self._forces_reference[livello7] = torch.zeros_like(self._forces_reference[livello7]).uniform_(19.0, 20.0)
        self._forces_reference[livello8] = torch.zeros_like(self._forces_reference[livello8]).uniform_(20.0, 20.0)
        

        # current material
        material = self._cuboid.root_physx_view.get_material_properties()

        # Levels
        material[livello0, 0, 2] = -1000.0
        material[livello1, 0, 2] = -1500.0
        material[livello2, 0, 2] = -2000.0
        material[livello3, 0, 2] = -2500.0
        material[livello4, 0, 2] = -3000.0
        material[livello5, 0, 2] = -3500.0
        material[livello6, 0, 2] = -4000.0
        material[livello7, 0, 2] = -4500.0
        material[livello8, 0, 2] = -5000.0
        material[livello9, 0, 2] = -5500.0
        material[livello10, 0, 2] = -6000.0
        material[livello11, 0, 2] = -6500.0
        material[livello12, 0, 2] = -7000.0
        material[livello13, 0, 2] = -7500.0
        material[livello14, 0, 2] = -8000.0
        material[livello15, 0, 2] = -8500.0
        material[livello16, 0, 2] = -9000.0
        material[livello17, 0, 2] = -9500.0
        material[livello18, 0, 2] = -10000.0
        material[livello19, 0, 2] = -10500.0
        material[livello20, 0, 2] = -11000.0
        material[livello21, 0, 2] = -11500.0
        material[livello22, 0, 2] = -12000.0
        material[livello23, 0, 2] = -12500.0
        material[livello24, 0, 2] = -13000.0
        material[livello25, 0, 2] = -13500.0
        material[livello26, 0, 2] = -14000.0
        material[livello27, 0, 2] = -14500.0
        material[livello28, 0, 2] = -15000.0
        material[livello29, 0, 2] = -15500.0
        material[livello30, 0, 2] = -16000.0
        material[livello31, 0, 2] = -16500.0
        material[livello32, 0, 2] = -17000.0
        material[livello33, 0, 2] = -17500.0
        material[livello34, 0, 2] = -18000.0
        material[livello35, 0, 2] = -18500.0
        material[livello36, 0, 2] = -19000.0
        material[livello37, 0, 2] = -19500.0
        material[livello38, 0, 2] = -20000.0
        material[livello39, 0, 2] = -20500.0
        material[livello40, 0, 2] = -21000.0
        material[livello41, 0, 2] = -21500.0
        material[livello42, 0, 2] = -22000.0
        material[livello43, 0, 2] = -22500.0
        material[livello44, 0, 2] = -23000.0
        material[livello45, 0, 2] = -23500.0
        material[livello46, 0, 2] = -24000.0
        material[livello47, 0, 2] = -24500.0
        material[livello48, 0, 2] = -25000.0
        env_ids_cup = env_ids.cpu()
        self._cuboid.root_physx_view.set_material_properties(material, env_ids_cup)

        if (self.count_int > 1000):
            self.count_int = 0
        
        self.count_int += 1
        if (self.count_int % 25 == 0):
            file_path = "/home/emanuele/dati.txt"
            with open(file_path, 'a') as file:
                file.write(f"Level:{self._level[env_ids]}\n")
                file.write(f"Count:{self.count}\n")


    
        self._forces_buffer[env_ids, :] = 0.0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)