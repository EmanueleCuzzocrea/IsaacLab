from __future__ import annotations
import gymnasium as gym
import torch
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
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
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
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

    #add_base_mass = EventTerm(
    #    func=mdp.randomize_rigid_body_mass,
    #    mode="startup",
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #        "mass_distribution_params": (-5.0, 5.0),
    #        "operation": "add",
    #    },
    #)

    ## reset
    #base_external_force_torque = EventTerm(
    #    func=mdp.apply_external_force_torque,
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #        "force_range": (-10.0, 10.0),
    #        "torque_range": (-10.0, 10.0),
    #    },
    #)

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
        physx=sim_utils.PhysxCfg(
            #solver_type=0,
            #enable_ccd=True,
            #bounce_threshold_velocity=100.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            #compliant_contact_stiffness=1000,
            #compliant_contact_damping=10000,
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
            #compliant_contact_stiffness=1000000,
            #compliant_contact_damping=1000,
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
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.0, track_air_time=True, track_pose=True,
    )
    
    cuboid_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10.0, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                compliant_contact_stiffness=1000,
                #compliant_contact_damping=3000,
                restitution=0.0,
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
    joint_accel_reward_scale = 0.0 #-2.5e-7
    joint_vel_reward_scale = 0.0 #-2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.0
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0*2

    track_yaw = 1.0
    force_variance = 0.0 #-0.000001
    force_acceleration = 0.0 #-0.001
    force_min_max = 0.0
    track_force = 0.0

    #lin_vel_reward_scale = 1.0*5
    #yaw_rate_reward_scale = 0.5*5
    #z_vel_reward_scale = -2.0*5
    #ang_vel_reward_scale = -0.05*8
    #joint_torque_reward_scale = -2.5e-8
    #joint_accel_reward_scale = -2.5e-7*6
    #action_rate_reward_scale = -0.01*6
    #feet_air_time_reward_scale = 0.0
    #undersired_contact_reward_scale = -1.0
    #flat_orientation_reward_scale = -5.0*3


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
        self._commands_b = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward machines parameters
        self._P = torch.zeros(self.num_envs, 4, device=self.device)
        self._state = torch.zeros(self.num_envs, 1, device=self.device)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._ok = torch.zeros(self.num_envs, 1, device=self.device)
        self._frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._sequenza_target_1 = torch.tensor([1, 1, 0, 0], device=self.device)
        self._sequenza_target_2 = torch.tensor([1, 1, 1, 1], device=self.device)
        self._sequenza_target_3 = torch.tensor([0, 0, 1, 1], device=self.device)
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
                "force_variance",
                "force_acceleration",
                "force_min_max",
                "force_tracking",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._interaction_ids, _ = self._contact_sensor.find_bodies("interaction")

        self._forces = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces_boolean = torch.zeros(self.num_envs, 1, device=self.device)
        self.yaw = torch.zeros(self.num_envs, 1, device=self.device)

        self._forces_buffer = torch.zeros(self.num_envs, 300, device=self.device)
        self._forces_buffer_normalized = torch.zeros(self.num_envs, 20, device=self.device)
        self._forces_filtered = torch.zeros(self.num_envs, 1, device=self.device)

        self._forces_reference = torch.zeros(self.num_envs, 1, device=self.device)

        self._forces_metric = torch.zeros(self.num_envs, 1, device=self.device)
        self._mae = torch.zeros(self.num_envs, 1, device=self.device)
        self._iteration = torch.zeros(self.num_envs, 1, device=self.device)

        self._extra_reward2 = torch.zeros(self.num_envs, 1, device=self.device)


        self.a = 0.0

        # Plot
        self.count = 0
        self.t_list = []
        self.force_list = []
        self.force_feet1_list = []
        self.force_feet2_list = []
        self.force_feet3_list = []
        self.force_feet4_list = []


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
        self.a += 0.0005*(50.0 - self._forces[0,0].item())
        self._forces_reference[:, 0] = 1.0
        if (self._forces[0,0].item() > 0.0):
            self._commands[:, 0] = 0.0
        else:
            self._commands[:, 0] = 0.2
        self._commands[:, 1] = 0.15
        self._commands[:, 2] = 0.0
        
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
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt*2)[:, self._feet_ids]
        first_air = self._contact_sensor.compute_first_air(self.step_dt*2)[:, self._feet_ids]

        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        mask_contact = current_contact_time > 0.0
        mask_air = current_air_time > 0.0
        self._P[:, :][mask_contact] = 1
        self._P[:, :][mask_air] = 0
        
        # reward machine
        self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        mask_phase = (self._phase[:, 0] < 16)
        self._phase[:, 0][mask_phase] += 1

        self._extra_reward2[:, 0] = 0

        maschera1_1 = (self._P == self._sequenza_target_1).all(dim=1) & (self._state == 0).all(dim=1)
        self._state[:, 0][maschera1_1] = 1
        self._ok[:, 0][maschera1_1] = 0
        maschera1_ok = (self._P != self._sequenza_target_1).any(dim=1) & (self._state == 1).all(dim=1)
        self._ok[:, 0][maschera1_ok] += 1
        maschera1_2 = (self._P == self._sequenza_target_1).all(dim=1) & (self._state == 1).all(dim=1) & (self._ok < 5).all(dim=1)
        self._state[:, 0][maschera1_2] = 1

        maschera2_1 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 1).all(dim=1)
        self._state[:, 0][maschera2_1] = 2
        self._ok[:, 0][maschera2_1] = 0
        maschera2_ok = (self._P != self._sequenza_target_2).any(dim=1) & (self._state == 2).all(dim=1)
        self._ok[:, 0][maschera2_ok] += 1
        maschera2_2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 2).all(dim=1) & (self._ok < 5).all(dim=1)
        self._state[:, 0][maschera2_2] = 2

        maschera3_1 = (self._P == self._sequenza_target_3).all(dim=1) & (self._state == 2).all(dim=1)
        self._state[:, 0][maschera3_1] = 3
        self._ok[:, 0][maschera3_1] = 0
        maschera3_ok = (self._P != self._sequenza_target_3).any(dim=1) & (self._state == 3).all(dim=1)
        self._ok[:, 0][maschera3_ok] += 1
        maschera3_2 = (self._P == self._sequenza_target_3).all(dim=1) & (self._state == 3).all(dim=1) & (self._ok < 5).all(dim=1)
        self._state[:, 0][maschera3_2] = 3

        maschera4_1 = (self._P == self._sequenza_target_4).all(dim=1) & (self._state == 3).all(dim=1)
        self._state[:, 0][maschera4_1] = 0
        self._ok[:, 0][maschera4_1] = 0
        maschera4_ok = (self._P != self._sequenza_target_4).any(dim=1) & (self._state == 0).all(dim=1)
        self._ok[:, 0][maschera4_ok] += 1
        maschera4_2 = (self._P == self._sequenza_target_4).all(dim=1) & (self._state == 0).all(dim=1) & (self._ok < 5).all(dim=1)
        self._state[:, 0][maschera4_2] = 0

        print("State", self._state)
        print("Ok", self._ok)

        

        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        
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
        #z_component *= torch.cos(self.yaw[:, 0])
        self._forces[:,0] = z_component
        #print(self._forces)
        #print(self._robot.data.root_lin_vel_b)
        
        # interaction force buffer
        self._forces_buffer[:, :-1] = self._forces_buffer[:, 1:].clone()
        self._forces_buffer[:, -1] = self._forces[:,0].squeeze()
       
        self.count += 1
        #print(self.count)
        self.t_list.append(self.count * (4/200))
        self.force_list.append(self._forces[0,0].item())
        #self.force_list.append(self._robot.data.root_lin_vel_b[0,0].item())
        

        # force tracking
        force_error = torch.square(self._forces_reference[:, 0] - self._forces[:, 0])
        force_error_mapped = torch.exp(-force_error / 0.25)
        
        # force variance
        force_variance = self._forces_buffer.var(dim=1)

        # force acceleration
        first_differences = self._forces_buffer[:, 1:] - self._forces_buffer[:, :-1]
        second_differences = first_differences[:, 1:] - first_differences[:, :-1]
        force_acceleration = second_differences.abs().mean(dim=1)

        # force min max
        force_min = self._forces_buffer.min(dim=1).values
        force_max = self._forces_buffer.max(dim=1).values
        force_min_max = force_max - force_min



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
            "force_variance": force_variance * self.cfg.force_variance * self.step_dt,
            "force_acceleration": force_acceleration * self.cfg.force_acceleration * self.step_dt,
            "force_min_max": force_min_max * self.cfg.force_min_max * self.step_dt,
            "force_tracking": force_error_mapped * self.cfg.track_force * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        #mask_extra = self._extra_reward > 0
        #reward[mask_extra] *= 2.0
#
        #mask_force = self._forces.squeeze(dim=1) > 0
        #reward[mask_force] += 0.001
        #print(reward)

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
        #if len(env_ids) == self.num_envs:
        #    # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        #    self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-0.5, 0.5)
    
        self._commands[env_ids, 2] = 0.0
        self._commands[env_ids,0] = 0.1
        self._commands[env_ids,1] = 0.0

        # Sample new force commands
        self._forces_reference[env_ids] = torch.zeros_like(self._forces_reference[env_ids]).uniform_(10.0, 10.0)
        self.a = 0.0
        self._forces_buffer[env_ids, :] = 0.0

        # Reward machines
        self._P[env_ids, :] = 0
        self._state[env_ids, 0] = 0
        self._phase[env_ids, 0] = 0
        self._ok[env_ids, 0] = 0
        self._frequency[env_ids, 0] = 3
      
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        
        cube_used = torch.tensor([1.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        self._cuboid.write_root_pose_to_sim(default_root_state[:, :7] + cube_used, env_ids)
    

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



        # plot
        if (self.count > 10):
            plt.figure(figsize=(10, 8))

            # force plot
            plt.plot(self.t_list, self.force_list, label='Interaction force')
            #plt.plot(self.t_list, self.force_feet1_list, label='Feet1 force')
            #plt.plot(self.t_list, self.force_feet2_list, label='Feet2 force')
            #plt.plot(self.t_list, self.force_feet3_list, label='Feet3 force')
            #plt.plot(self.t_list, self.force_feet4_list, label='Feet4 force')
            plt.title('Interaction force', fontsize=28)
            plt.xlabel('Time [s]', fontsize=28)
            plt.ylabel('Force [N]', fontsize=28)
            plt.grid()
            plt.tick_params(axis='both', which='major', labelsize=25) 
            plt.legend()

            # show plots
            plt.tight_layout()
            plt.show()