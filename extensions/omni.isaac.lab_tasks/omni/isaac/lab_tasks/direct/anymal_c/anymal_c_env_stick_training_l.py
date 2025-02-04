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
from omni.isaac.lab_assets.anymal import ANYMAL_STICK_L_CFG  # isort: skip
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

    #reset_base = EventTerm(
    #    func=mdp.reset_root_state_uniform,
    #    mode="reset",
    #    params={
    #        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #        "velocity_range": {
    #            "x": (-0.5, 0.5),
    #            "y": (-0.5, 0.5),
    #            "z": (-0.5, 0.5),
    #            "roll": (-0.5, 0.5),
    #            "pitch": (-0.5, 0.5),
    #            "yaw": (-0.5, 0.5),
    #        },
    #    },
    #)

    #velocity_base = EventTerm(
    #    func=mdp.push_by_setting_velocity,
    #    mode="reset",
    #    is_global_time=True,
    #    interval_range_s=(0.0,0.0),
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #        "velocity_range": {
    #            #"x": (0.0, 0.0),
    #            #"y": (0., 0.0),
    #            #"z": (-0.0, 0.0),
    #            #"roll": (-0.0, 0.0),
    #            #"pitch": (-0.0, 0.0),
    #            #"yaw": (-0.0, 0.0),
    #        },
    #    },
    #)


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
    robot: ArticulationCfg = ANYMAL_STICK_L_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True, track_pose=True,
    )

    chunk_4 = "\d\d\d[02468]"
    cuboid_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_{chunk_4}/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 0.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=100000,
                compliant_contact_damping=1000,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.9, 0.8)),
    )


    # reward scales
    lin_vel_reward_scale_x = 1.5
    lin_vel_reward_scale_y = 1.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05*3
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    joint_vel_reward_scale = 0.0
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.0
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0*2

    track_yaw = 1.5
    track_force = 0.0
    track_force2 = 2.0
    joint_deviation = -0.7
    energy = -0.00005



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
        self._ok = torch.zeros(self.num_envs, 1, device=self.device)
        self._frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        self._extra_reward2 = torch.zeros(self.num_envs, 1, device=self.device)
        self._extra_reward3 = torch.zeros(self.num_envs, 1, device=self.device)
        self._transition_cost = torch.zeros(self.num_envs, 1, device=self.device)
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
                "energy",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._interaction_ids, _ = self._contact_sensor.find_bodies("interaction")

        self.yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces_reference = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces_buffer = torch.zeros(self.num_envs, 15, device=self.device)

        self.count = 0.0
        self.count_int = 0
        self.t_min = 5
        self.epsilon = 0.3

        self.counter = torch.zeros(self.num_envs, 1, device=self.device)
        self.count_4 = torch.zeros(self.num_envs, 1, device=self.device)
        

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
        mask_p = (torch.arange(4096, device=self.device) >= 1000) & \
            (torch.arange(4096, device=self.device) % 2 == 0)
        mask_d = (torch.arange(4096, device=self.device) < 1000) | \
            (torch.arange(4096, device=self.device) % 2 != 0)

        self._commands[mask_p, 0] = 0.1
        self._commands[mask_p, 1] = -0.2
        mask_force__ = self._forces.squeeze(dim=1) > 0.0001
        self._commands[mask_force__, 1] = 0.0

        self.counter[:, 0] += 1

        mask_change_vel_d = (self.counter[:, 0] > 200) & mask_d
        self.counter[mask_change_vel_d, 0] = 0
        self._commands[mask_change_vel_d, 0] = torch.zeros_like(self._commands[mask_change_vel_d, 0]).uniform_(-1.0, 1.0)
        self._commands[mask_change_vel_d, 1] = torch.zeros_like(self._commands[mask_change_vel_d, 1]).uniform_(-1.0, 1.0)
        
        
        mask1 = (self._commands[:, 0] > 0.7)
        self._commands[mask1, 0] = 0.0
        mask2 = (self._commands[:, 0] < -0.7)
        self._commands[mask2, 0] = 0.0
        mask3 = (self._commands[:, 1] > 0.7)
        self._commands[mask3, 1] = 0.0
        mask4 = (self._commands[:, 1] < -0.7)
        self._commands[mask4, 1] = 0.0
    
        mask5 = (self._commands[:, 0] < 0.05) & (self._commands[:, 0] > -0.05)
        self._commands[mask5, 0] = 0.0
        mask6 = (self._commands[:, 1] < 0.05) & (self._commands[:, 1] > -0.05)
        self._commands[mask6, 1] = 0.0


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
                    #self._phase,
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
        yaw_error_mapped = torch.exp(-yaw_error / 0.15)

        # linear velocity tracking_x
        lin_vel_error_x = torch.square(self._commands[:, 0] - self._robot.data.root_lin_vel_b[:, 0])
        lin_vel_error_mapped_x = torch.exp(-lin_vel_error_x / 0.15)
        # linear velocity tracking_y
        lin_vel_error_y = torch.square(self._commands[:, 1] - self._robot.data.root_lin_vel_b[:, 1])
        lin_vel_error_mapped_y = torch.exp(-lin_vel_error_y / 0.15)
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
        # energy
        energy = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        first_air = self._contact_sensor.compute_first_air(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 1.0) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )


        # reward machine2
        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        mask_contact = current_contact_time > 0.0
        mask_air = current_air_time > 0.0
        self._P[:, :][mask_contact] = 1
        self._P[:, :][mask_air] = 0

        mask_phase = (self._phase[:, 0] < 19)
        self._phase[:, 0][mask_phase] += 1
        
        self._extra_reward2[:, 0] = 0
        self._extra_reward3[:, 0] = 0
        self._transition_cost[:, 0] = 0


        maschera1_1 = (self._P == self._sequenza_target_1).all(dim=1) & (self._state == 0).all(dim=1)
        self._state[:, 0][maschera1_1] = 1
        self._ok[:, 0][maschera1_1] = 0
        self._transition_cost[:, 0][maschera1_1] = 1
        self._phase[:, 0][maschera1_1] = 0
        maschera1_ok = (self._P != self._sequenza_target_1).any(dim=1) & (self._state == 1).all(dim=1)
        self._ok[:, 0][maschera1_ok] += 1
        maschera1_2 = (self._P == self._sequenza_target_1).all(dim=1) & (self._state == 1).all(dim=1) & (self._ok < 1).all(dim=1)
        self._extra_reward3[:, 0][maschera1_2] = 1


        maschera2_1 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 1).all(dim=1)
        self._state[:, 0][maschera2_1] = 2
        self._ok[:, 0][maschera2_1] = 0
        self._transition_cost[:, 0][maschera2_1] = 1
        self._phase[:, 0][maschera2_1] = 0
        maschera2_ok = (self._P != self._sequenza_target_2).any(dim=1) & (self._state == 2).all(dim=1)
        self._ok[:, 0][maschera2_ok] += 1
        maschera2_2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 2).all(dim=1) & (self._ok < 1).all(dim=1)
        self._extra_reward2[:, 0][maschera2_2] = 1


        maschera3_1 = (self._P == self._sequenza_target_3).all(dim=1) & (self._state == 2).all(dim=1)
        self._state[:, 0][maschera3_1] = 3
        self._ok[:, 0][maschera3_1] = 0
        self._transition_cost[:, 0][maschera3_1] = 1
        self._phase[:, 0][maschera3_1] = 0
        maschera3_ok = (self._P != self._sequenza_target_3).any(dim=1) & (self._state == 3).all(dim=1)
        self._ok[:, 0][maschera3_ok] += 1
        maschera3_2 = (self._P == self._sequenza_target_3).all(dim=1) & (self._state == 3).all(dim=1) & (self._ok < 1).all(dim=1)
        self._extra_reward3[:, 0][maschera3_2] = 1


        maschera4_1 = (self._P == self._sequenza_target_4).all(dim=1) & (self._state == 3).all(dim=1)
        self._state[:, 0][maschera4_1] = 0
        self._ok[:, 0][maschera4_1] = 0
        self._transition_cost[:, 0][maschera4_1] = 1
        self._phase[:, 0][maschera4_1] = 0
        maschera4_ok = (self._P != self._sequenza_target_4).any(dim=1) & (self._state == 0).all(dim=1)
        self._ok[:, 0][maschera4_ok] += 1
        maschera4_2 = (self._P == self._sequenza_target_4).all(dim=1) & (self._state == 0).all(dim=1) & (self._ok < 1).all(dim=1)
        self._extra_reward2[:, 0][maschera4_2] = 1


        self._extra_reward2_ = self._extra_reward2.squeeze()
        self._extra_reward3_ = self._extra_reward3.squeeze()
        self._transition_cost_ = self._transition_cost.squeeze()

        #self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        #maschera1 = (self._P == self._sequenza_target_1).all(dim=1) & (self._phase > 0).all(dim=1) & (self._state == 0).all(dim=1)
        #self._state[:, 0][maschera1] = 1
        #self._extra_reward[maschera1] = 2
        #self._phase[:, 0][maschera1] = 0
        #maschera2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._phase > 5).all(dim=1) & (self._state == 1).all(dim=1)
        #self._state[:, 0][maschera2] = 2
        #self._extra_reward[maschera2] = 2
        #self._phase[:, 0][maschera2] = 0
        #maschera3 = (self._P == self._sequenza_target_3).all(dim=1) & (self._phase > 0).all(dim=1) & (self._state == 2).all(dim=1)
        #self._state[:, 0][maschera3] = 3
        #self._extra_reward[maschera3] = 2
        #self._phase[:, 0][maschera3] = 0
        #maschera4 = (self._P == self._sequenza_target_4).all(dim=1) & (self._phase > 5).all(dim=1) & (self._state == 3).all(dim=1)
        #self._state[:, 0][maschera4] = 0
        #self._extra_reward[maschera4] = 2
        #self._phase[:, 0][maschera4] = 0
        #self._extra_reward = self._extra_reward.squeeze()


    
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
        z_component = torch.abs(interaction_force[:, 1])
        #z_component *= torch.cos(self.yaw[:, 0])
        self._forces[:,0] = z_component


        # interaction force buffer
        self._forces_buffer[:, :-1] = self._forces_buffer[:, 1:].clone()
        self._forces_buffer[:, -1] = self._forces[:,0].squeeze()


        # force tracking
        force_error = torch.square(self._forces_reference[:, 0] - self._forces[:, 0])
        force_error_mapped = torch.exp(-force_error / 25)
        mask_ref = self._forces_reference[:, 0] < 0.5
        force_error[mask_ref] = 0.0
        force_error_mapped[mask_ref] = 0.0
        

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
            "energy": energy * self.cfg.energy * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)


        mask_extra3_ = (self._extra_reward3_ > 0.5)
        reward[mask_extra3_] *= 1.5

        mask_extra2_ = (self._extra_reward2_ > 0.5)
        reward[mask_extra2_] *= 1.5
        
        #mask_cost = (self._transition_cost_ > 0.5)
        #reward[mask_cost] /= 1.2

        #mask_extra = (self._extra_reward > 0.5)
        #reward[mask_extra] *= 2.0


        #if (self.count_int > 1000):
        #    self.count_int = 0
        #
        #self.count_int += 1
        #if (self.count_int % 25 == 0):
        #    file_path = "/home/emanuele/dati.txt"
        #    with open(file_path, 'w') as file:
        #        file.write(f"self.count_int: {self.count_int}\n")
        #        file.write(f"self.t_min: {self.t_min}\n")

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

        dispari = env_ids[(env_ids % 2 != 0) | (env_ids < 1000)]
        pari = env_ids[(env_ids % 2 == 0) & (env_ids > 999)]
        
        # Sample new commands
        self._commands[env_ids, 2] = 0.0
            
        x_ = random.uniform(-0.2, 0.2)
        self._commands[pari,0] = x_


        # Sample new force commands
        self._forces_reference[pari] = torch.zeros_like(self._forces_reference[pari]).uniform_(40.0, 40.0)
        self._forces_buffer[env_ids, :] = 0.0


        # Reward machines
        self._P[env_ids, :] = 0
        self._state[env_ids, 0] = 0
        self._phase[env_ids, 0] = 0
        self._ok[env_ids, 0] = 0
        self._frequency[dispari, 0] = random.randint(1, 2)
        self._frequency[pari, 0] = random.randint(1, 5)

        # change velocity
        self.counter[env_ids, :] = 0
        self.count_4[env_ids, :] = 0

      
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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