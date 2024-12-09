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
            "static_friction_range": (0.6, 0.8),
            "dynamic_friction_range": (0.4, 0.6),
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
    robot: ArticulationCfg = ANYMAL_STICK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True, track_pose=True,
    )

    chunk_4 = "\d\d\d[02468]"
    cuboid_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=250,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    )

    chunk_4 = "\d\d\d[02468]"
    cuboid_cfg1: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=500,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.7137, 0.7569), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 20.0)),
    )
    chunk_4 = "\d\d\d[02468]"
    cuboid_cfg2: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=1000,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6784, 0.8471, 0.9020), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 25.0)),
    )

    chunk_4 = "\d\d\d[02468]"
    cuboid_cfg3: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cuboid3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.5, 1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                compliant_contact_stiffness=2000,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8471, 0.7490, 0.8471), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 30.0)),
    )

    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg4: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid4",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=5000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8549, 0.7255), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)
#
    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg5: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid5",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=6000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5961, 0.9843, 0.5961), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)
#
    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg6: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid6",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=7000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9020, 0.9020, 0.9804), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)
#
    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg7: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid7",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=8000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9922, 0.9922, 0.5882), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)
#
    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg8: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid8",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=9000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7412, 0.7176, 0.4196), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)
#
    #chunk_4 = "\d\d\d[02468]"
    #cuboid_cfg9: RigidObjectCfg = RigidObjectCfg(
    #    prim_path=f"/World/envs/env_.*/Cuboid9",
    #    spawn=sim_utils.CuboidCfg(
    #        size=(0.5, 4.0, 2.0),
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            static_friction=0.5,
    #            dynamic_friction=0.5,
    #            compliant_contact_stiffness=10000,
    #            restitution=0.0,
    #        ),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            disable_gravity=True,
    #        ),
    #        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9412, 0.5020, 0.5020), metallic=0.2),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 10.0)),
    #)


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
                "force_variance",
                "force_acceleration",
                "force_min_max",
                "force_tracking",
                "force_tracking2",
                "force_jerk",
                "force_snap",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._interaction_ids, _ = self._contact_sensor.find_bodies("interaction")

        self._forces = torch.zeros(self.num_envs, 1, device=self.device)
        self.yaw = torch.zeros(self.num_envs, 1, device=self.device)

        self._forces_buffer = torch.zeros(self.num_envs, 300, device=self.device)
        self._forces_filtered = torch.zeros(self.num_envs, 1, device=self.device)
        self._forces_data = torch.zeros(self.num_envs, 4, device=self.device)

        self._forces_reference = torch.zeros(self.num_envs, 1, device=self.device)
        self._integrators = torch.zeros(self.num_envs, 1, device=self.device)
        self._proportional = torch.zeros(self.num_envs, 1, device=self.device)

        self._forces_metric = torch.zeros(self.num_envs, 1, device=self.device)
        self._mae = torch.zeros(self.num_envs, 1, device=self.device)
        self._iteration = torch.zeros(self.num_envs, 1, device=self.device)

        self._level = torch.zeros(self.num_envs, 1, device=self.device)
        self.count = 0

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
        self._cuboid1 = RigidObject(self.cfg.cuboid_cfg1)
        self._cuboid2 = RigidObject(self.cfg.cuboid_cfg2)
        self._cuboid3 = RigidObject(self.cfg.cuboid_cfg3)
        #self._cuboid4 = RigidObject(self.cfg.cuboid_cfg4)
        #self._cuboid5 = RigidObject(self.cfg.cuboid_cfg5)
        #self._cuboid6 = RigidObject(self.cfg.cuboid_cfg6)
        #self._cuboid7 = RigidObject(self.cfg.cuboid_cfg7)
        #self._cuboid8 = RigidObject(self.cfg.cuboid_cfg8)
        #self._cuboid9 = RigidObject(self.cfg.cuboid_cfg9)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)        

    def _get_observations(self) -> dict:
        mask = (torch.arange(4096, device=self.device) >= 1000) & \
            (torch.arange(4096, device=self.device) <= 4095) & \
            (torch.arange(4096, device=self.device) % 2 == 0)

        self._integrators[mask, 0] += 0.0003*(self._forces_reference[mask, 0] - self._forces[mask, 0])
        self._proportional[mask, 0] = 0.0003*(self._forces_reference[mask, 0] - self._forces[mask, 0])
        self._commands[mask, 0] =  self._integrators[mask, 0] #+ self._proportional[mask, 0]

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
                    self._frequency,
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
        first_air = self._contact_sensor.compute_first_air(self.step_dt)[:, self._feet_ids]

        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        mask_contact = current_contact_time > 0.0
        mask_air = current_air_time > 0.0
        self._P[:, :][mask_contact] = 1
        self._P[:, :][mask_air] = 0
        
        # reward machine
        self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        #mask_phase1 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 1).all(dim=1) & (self._phase[:, 0] < 16)
        #mask_phase2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._state == 3).all(dim=1) & (self._phase[:, 0] < 16)
        #mask_phase = mask_phase1 | mask_phase2
        mask_phase = (self._phase[:, 0] < 16)
        self._phase[:, 0][mask_phase] += 1

        maschera1 = (self._P == self._sequenza_target_1).all(dim=1) & (self._phase > self._frequency).all(dim=1) & (self._state == 0).all(dim=1)
        self._state[:, 0][maschera1] = 1
        self._extra_reward[maschera1] = 2
        self._phase[:, 0][maschera1] = 0
        maschera2 = (self._P == self._sequenza_target_2).all(dim=1) & (self._phase > self._frequency).all(dim=1) & (self._state == 1).all(dim=1)
        self._state[:, 0][maschera2] = 2
        self._extra_reward[maschera2] = 2
        self._phase[:, 0][maschera2] = 0
        maschera3 = (self._P == self._sequenza_target_3).all(dim=1) & (self._phase > self._frequency).all(dim=1) & (self._state == 2).all(dim=1)
        self._state[:, 0][maschera3] = 3
        self._extra_reward[maschera3] = 2
        self._phase[:, 0][maschera3] = 0
        maschera4 = (self._P == self._sequenza_target_4).all(dim=1) & (self._phase > self._frequency).all(dim=1) & (self._state == 3).all(dim=1)
        self._state[:, 0][maschera4] = 0
        self._extra_reward[maschera4] = 2
        self._phase[:, 0][maschera4] = 0
        self._extra_reward = self._extra_reward.squeeze()

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
        z_component *= torch.cos(self.yaw[:, 0])
        self._forces[:,0] = z_component

        # interaction force buffer
        self._forces_buffer[:, :-1] = self._forces_buffer[:, 1:].clone()
        self._forces_buffer[:, -1] = self._forces[:,0].squeeze()



        # force tracking
        force_error = torch.square(self._forces_reference[:, 0] - self._forces[:, 0])
        force_error_mapped = torch.exp(-force_error / 0.25)
        
        # force variance
        force_variance = self._forces_buffer.var(dim=1)

        # force velocity
        first_differences = self._forces_buffer[:, 1:] - self._forces_buffer[:, :-1]
        force_velocity = first_differences.abs().mean(dim=1)

        # force acceleration
        second_differences = first_differences[:, 1:] - first_differences[:, :-1]
        force_acceleration = second_differences.abs().mean(dim=1)

        # force jerk
        third_difference = second_differences[:, 1:] - second_differences[:, :-1]
        force_jerk = third_difference.abs().mean(dim=1)

        # force snap
        forth_difference = third_difference[:, 1:] - third_difference[:, :-1]
        force_snap = forth_difference.abs().mean(dim=1)

        # force min max
        force_min = self._forces_buffer.min(dim=1).values
        force_max = self._forces_buffer.max(dim=1).values
        force_min_max = force_max - force_min

        # force data
        self._forces_data[:, 0] = force_variance
        self._forces_data[:, 1] = force_acceleration
        self._forces_data[:, 2] = force_min
        self._forces_data[:, 3] = force_max

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
            "force_tracking": force_error * self.cfg.track_force * self.step_dt,
            "force_tracking2": force_error_mapped * self.cfg.track_force2 * self.step_dt,
            "force_jerk": force_jerk * self.cfg.force_jerk * self.step_dt,
            "force_snap": force_snap * self.cfg.force_snap * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        mask_extra = self._extra_reward > 0
        #reward[mask_extra] += 0.15
        reward[mask_extra] *= 2.0

        #mask_force = self._forces.squeeze(dim=1) > 0.0
        #reward[mask_force] += 0.001

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
        #self._cuboid.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-0.5, 0.5)

        dispari = env_ids[(env_ids % 2 != 0) | (env_ids < 1000)]
        pari = env_ids[(env_ids % 2 == 0) & (env_ids > 999)]

        yaw_ = random.uniform(-3.14, 3.14)
        self._commands[dispari, 2] = yaw_
        numero = random.randint(1, 10)
        if numero == 5:
            self._commands[dispari,0] *= 0.0
            self._commands[dispari,1] *= 0.0
    
        self._commands[pari, 2] = 0.0
        x_ = random.uniform(0.0, 0.3)
        y_ = random.uniform(-0.2, 0.2)
        self._commands[pari,0] = 0.0
        self._commands[pari,1] = 0.0

        # Sample new force commands
        self._forces_reference[pari] = torch.zeros_like(self._forces_reference[pari]).uniform_(10.0, 10.0)
        self._integrators[env_ids] = 0.0
        


        # Reward machines
        self._P[env_ids, :] = 0
        self._state[env_ids, 0] = 0
        self._phase[env_ids, 0] = 0
        self._frequency[dispari, 0] = random.randint(2, 4)
        self._frequency[pari, 0] = random.randint(2, 4)

      
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        # Curriculum learning
        #self._cuboid.reset(env_ids)
        #self._cuboid1.reset(env_ids)
        mask_level_up = (self._forces_buffer.min(dim=1).values > 7) & (self._forces_buffer.max(dim=1).values < 13)
        selected_mask_up = mask_level_up[env_ids]
        self._level[env_ids[selected_mask_up], 0] += 1

        mask_level_down = (self._forces_buffer.min(dim=1).values < 5) | (self._forces_buffer.max(dim=1).values > 25)
        selected_mask_down = mask_level_down[env_ids]
        self._level[env_ids[selected_mask_down], 0] -= 1
        self._level[:, 0].clamp_(min=0)

        cube_used = torch.tensor([1.15, 0.0, 0.0], device=self.device)
        cube_not_used = torch.tensor([1.15, 0.0, 10.0], device=self.device)
        cube_not_used2 = torch.tensor([1.15, 0.0, 20.0], device=self.device)
        cube_not_used3 = torch.tensor([1.15, 0.0, 25.0], device=self.device)

        livello0 = env_ids[(self._level[env_ids, 0] == 0) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello1 = env_ids[(self._level[env_ids, 0] == 1) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello2 = env_ids[(self._level[env_ids, 0] == 2) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello3 = env_ids[(self._level[env_ids, 0] == 3) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello4 = env_ids[(self._level[env_ids, 0] == 4) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello5 = env_ids[(self._level[env_ids, 0] == 5) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello6 = env_ids[(self._level[env_ids, 0] == 6) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello7 = env_ids[(self._level[env_ids, 0] == 7) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello8 = env_ids[(self._level[env_ids, 0] == 8) & (env_ids % 2 == 0) & (env_ids > 999)]
        livello9 = env_ids[(self._level[env_ids, 0] == 9) & (env_ids % 2 == 0) & (env_ids > 999)]

        # Livello 0
        default_root_state_livello0 = self._robot.data.default_root_state[livello0]
        default_root_state_livello0[:, :3] += self._terrain.env_origins[livello0]
        cube_pose_used_livello0 = default_root_state_livello0.clone()
        cube_pose_not_used_livello0 = default_root_state_livello0.clone()
        cube_pose_not_used_livello0_2 = default_root_state_livello0.clone()
        cube_pose_not_used_livello0_3 = default_root_state_livello0.clone()
        cube_pose_used_livello0[:, :3] += cube_used
        cube_pose_not_used_livello0[:, :3] += cube_not_used
        cube_pose_not_used_livello0_2[:, :3] += cube_not_used2
        cube_pose_not_used_livello0_3[:, :3] += cube_not_used3
        self._cuboid.write_root_pose_to_sim(cube_pose_used_livello0[:, :7], livello0)
        self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello0_2[:, :7], livello0)
        self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello0_3[:, :7], livello0)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello0[:, :7], livello0)
        
        ## Livello 1
        default_root_state_livello1 = self._robot.data.default_root_state[livello1]
        default_root_state_livello1[:, :3] += self._terrain.env_origins[livello1]
        cube_pose_used_livello1 = default_root_state_livello1.clone()
        cube_pose_not_used_livello1 = default_root_state_livello1.clone()
        cube_pose_not_used_livello1_2 = default_root_state_livello1.clone()
        cube_pose_not_used_livello1_3 = default_root_state_livello1.clone()
        cube_pose_used_livello1[:, :3] += cube_used
        cube_pose_not_used_livello1[:, :3] += cube_not_used
        cube_pose_not_used_livello1_2[:, :3] += cube_not_used2
        cube_pose_not_used_livello1_3[:, :3] += cube_not_used3
        self._cuboid.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        self._cuboid1.write_root_pose_to_sim(cube_pose_used_livello1[:, :7], livello1)
        self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello1_2[:, :7], livello1)
        self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello1_3[:, :7], livello1)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello1[:, :7], livello1)
        
        # Livello 2
        default_root_state_livello2 = self._robot.data.default_root_state[livello2]
        default_root_state_livello2[:, :3] += self._terrain.env_origins[livello2]
        cube_pose_used_livello2 = default_root_state_livello2.clone()
        cube_pose_not_used_livello2 = default_root_state_livello2.clone()
        cube_pose_not_used_livello2_2 = default_root_state_livello2.clone()
        cube_pose_not_used_livello2_3 = default_root_state_livello2.clone()
        cube_pose_used_livello2[:, :3] += cube_used
        cube_pose_not_used_livello2[:, :3] += cube_not_used
        cube_pose_not_used_livello2_2[:, :3] += cube_not_used2
        cube_pose_not_used_livello2_3[:, :3] += cube_not_used3
        self._cuboid.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello2_2[:, :7], livello2)
        self._cuboid2.write_root_pose_to_sim(cube_pose_used_livello2[:, :7], livello2)
        self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello2_3[:, :7], livello2)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello2[:, :7], livello2)

        # Livello 3
        default_root_state_livello3 = self._robot.data.default_root_state[livello3]
        default_root_state_livello3[:, :3] += self._terrain.env_origins[livello3]
        cube_pose_used_livello3 = default_root_state_livello3.clone()
        cube_pose_not_used_livello3 = default_root_state_livello3.clone()
        cube_pose_not_used_livello3_2 = default_root_state_livello3.clone()
        cube_pose_not_used_livello3_3 = default_root_state_livello3.clone()
        cube_pose_used_livello3[:, :3] += cube_used
        cube_pose_not_used_livello3[:, :3] += cube_not_used
        cube_pose_not_used_livello3_2[:, :3] += cube_not_used2
        cube_pose_not_used_livello3_3[:, :3] += cube_not_used3
        self._cuboid.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello3_2[:, :7], livello3)
        self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello3_3[:, :7], livello3)
        self._cuboid3.write_root_pose_to_sim(cube_pose_used_livello3[:, :7], livello3)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello3[:, :7], livello3)
#
        ## Livello 4
        #default_root_state_livello4 = self._robot.data.default_root_state[livello4]
        #default_root_state_livello4[:, :3] += self._terrain.env_origins[livello4]
        #cube_pose_used_livello4 = default_root_state_livello4.clone()
        #cube_pose_not_used_livello4 = default_root_state_livello4.clone()
        #cube_pose_used_livello4[:, :3] += cube_used
        #cube_pose_not_used_livello4[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello4[:  :7], livello4)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello4[:, :7], livello4)
        #
        ## Livello 5
        #default_root_state_livello5 = self._robot.data.default_root_state[livello5]
        #default_root_state_livello5[:, :3] += self._terrain.env_origins[livello5]
        #cube_pose_used_livello5 = default_root_state_livello5.clone()
        #cube_pose_not_used_livello5 = default_root_state_livello5.clone()
        #cube_pose_used_livello5[:, :3] += cube_used
        #cube_pose_not_used_livello5[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello5[:  :7], livello5)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello5[:, :7], livello5)
#
        ## Livello 6
        #default_root_state_livello6 = self._robot.data.default_root_state[livello6]
        #default_root_state_livello6[:, :3] += self._terrain.env_origins[livello6]
        #cube_pose_used_livello6 = default_root_state_livello6.clone()
        #cube_pose_not_used_livello6 = default_root_state_livello6.clone()
        #cube_pose_used_livello6[:, :3] += cube_used
        #cube_pose_not_used_livello6[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello6[:  :7], livello6)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello6[:, :7], livello6)
#
        ## Livello 7
        #default_root_state_livello7 = self._robot.data.default_root_state[livello7]
        #default_root_state_livello7[:, :3] += self._terrain.env_origins[livello7]
        #cube_pose_used_livello7 = default_root_state_livello7.clone()
        #cube_pose_not_used_livello7 = default_root_state_livello7.clone()
        #cube_pose_used_livello7[:, :3] += cube_used
        #cube_pose_not_used_livello7[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello7[:  :7], livello7)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello7[:, :7], livello7)
#
        ## Livello 8
        #default_root_state_livello8 = self._robot.data.default_root_state[livello8]
        #default_root_state_livello8[:, :3] += self._terrain.env_origins[livello8]
        #cube_pose_used_livello8 = default_root_state_livello8.clone()
        #cube_pose_not_used_livello8 = default_root_state_livello8.clone()
        #cube_pose_used_livello8[:, :3] += cube_used
        #cube_pose_not_used_livello8[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello8[:  :7], livello8)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello8[:, :7], livello8)
#
        ## Livello 9
        #default_root_state_livello9 = self._robot.data.default_root_state[livello9]
        #default_root_state_livello9[:, :3] += self._terrain.env_origins[livello9]
        #cube_pose_used_livello9 = default_root_state_livello9.clone()
        #cube_pose_not_used_livello9 = default_root_state_livello9.clone()
        #cube_pose_used_livello9[:, :3] += cube_used
        #cube_pose_not_used_livello9[:, :3] += cube_not_used
        #self._cuboid.write_root_pose_to_sim(cube_pose_used_livello9[:  :7], livello9)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid2.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid3.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid4.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid5.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid6.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid7.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid8.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)
        #self._cuboid9.write_root_pose_to_sim(cube_pose_not_used_livello9[:, :7], livello9)

        #default_root_state_dispari = self._robot.data.default_root_state[dispari]
        #default_root_state_dispari[:, :3] += self._terrain.env_origins[dispari]
        #cube_pose_used_dispari = default_root_state_dispari.clone()
        #cube_pose_not_used_dispari = default_root_state_dispari.clone()
        #cube_pose_not_used_dispari2 = default_root_state_dispari.clone()
        #cube_pose_used_dispari[:, :3] += cube_used
        #cube_pose_not_used_dispari[:, :3] += cube_not_used
        #cube_pose_not_used_dispari2[:, :3] += cube_not_used2
        #self._cuboid.write_root_pose_to_sim(cube_pose_not_used_dispari[:, :7], dispari)
        #self._cuboid1.write_root_pose_to_sim(cube_pose_not_used_dispari2[:, :7], dispari)




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