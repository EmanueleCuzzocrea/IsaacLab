from __future__ import annotations
import gymnasium as gym
import torch
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
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
# Pre-defined configs
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip
from omni.isaac.lab_assets.anymal import ANYMAL_KINOVA_CFG  # isort: skip
from omni.isaac.lab_assets.anymal import ANYMAL_STICK_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# mpc
import numpy as np
import os
import rospkg
import matplotlib.pyplot as plt
from ocs2_mobile_manipulator import mpc_interface
from ocs2_mobile_manipulator import (
    scalar_array,
    vector_array,
    TargetTrajectories,
)
from MobileManipulatorPyBindingTest import compute_mpc_control

# Path of the package
packageDir = rospkg.RosPack().get_path('ocs2_mobile_manipulator')
taskFile = os.path.join(packageDir, 'config/stick/task.info')
libFolder = os.path.join(packageDir, 'auto_generated')
urdfDir = rospkg.RosPack().get_path('ocs2_robotic_assets')
urdf_ = os.path.join(urdfDir, 'resources/mobile_manipulator/stick/urdf/stick.urdf')

# Initialize MPC interface
mpc = mpc_interface(taskFile, libFolder, urdf_)

# State and input dimensions
stateDim = 7
inputDim = 7

# Markers
frame_marker_cfg = FRAME_MARKER_CFG.copy()
frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

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


@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    #episode_length_s = 100.0
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
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0


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
        self._P = torch.zeros(self.num_envs, 6, device=self.device)
        self._sequenza_target_1 = torch.tensor([0, 1, 1, 0, 0, 0], device=self.device)
        self._sequenza_target_2 = torch.tensor([0, 0, 0, 1, 1, 1], device=self.device)
        #self._sequenza_target_3 = torch.tensor([0, 1, 1, 0, 1, 2], device=self.device)
        #self._sequenza_target_4 = torch.tensor([0, 1, 1, 1, 0, 3], device=self.device)

        # Target pose
        self.target_pos = torch.zeros(1, 3, device=self.device)
        self.target_or = torch.zeros(1, 3, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._interaction_ids, _ = self._contact_sensor.find_bodies("interaction")

        #self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        #self._forces[0,0,1] = 100
        #self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Admittance filter
        self.m = 2
        self.k = 500
        self.d = 20.0
        self.dt_ = 4/200

        # Matrici del sistema nello spazio degli stati
        self.A = np.block([
            [0.0, 1.0],
            [-self.k/self.m, -self.d/self.m]
        ])
        self.B = np.vstack([0.0, 1/self.m])

        # Stato iniziale [Delta_p, Delta_p_dot]
        self.x = np.zeros((2,))
        self.f_x = np.zeros((1,))

        # Plot
        self.count = 0
        self.t_list = []
        self.force_list = []

        self._forces = torch.zeros(self.num_envs, 3, device=self.device)

        self._h = 0.0

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
        
        # add a cuboid
        self._cuboid_cfg = sim_utils.CuboidCfg(
            size=(0.5, 4, 1),
            #rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=0.2),
        )
        self._cuboid_cfg.func("/World/envs/env_.*/Cone", self._cuboid_cfg, translation=(3.15, 0.0, 0.5))



    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # interaction force
        interaction_force = self._contact_sensor.data.net_forces_w[:, self._interaction_ids]
        self.f_x[0] = interaction_force[0][0][0].item()

        # admittance filter
        x_dot = self.A @ self.x + self.B @ (self.f_x)
        self.x = self.x + self.dt_*x_dot

        self._h += 0.001

        # reset mpc
        desiredTimeTraj = scalar_array()
        desiredTimeTraj.push_back(0.0)
        desiredInputTraj = vector_array()
        desiredInputTraj.push_back(np.zeros(inputDim))
        desiredStateTraj = vector_array()
        #desiredStateTraj.push_back(np.array([3.0+self.x[0], 0.0, 0.6, 0.0, 0.0, 0.7, 0.7]))
        desiredStateTraj.push_back(np.array([3.0, self._h, 0.6, 0.0, 0.0, 0.7, 0.7]))
        targetTrajectories = TargetTrajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)
        mpc.reset(targetTrajectories)

        #self.target_pos = torch.tensor([[3.0+self.x[0], 0.0, 0.75]], device=self.device)
        self.target_pos = torch.tensor([[3.0, self._h, 0.75]], device=self.device)
        self.target_or = torch.tensor([[0.7, 0.0, 0.0, 0.7]], device=self.device)
        ee_marker.visualize(self.target_pos, self.target_or)


        root_pos_ = self._robot.data.root_pos_w
        root_quat_ = self._robot.data.root_quat_w
        w = root_quat_[0][0].item()
        x = root_quat_[0][1].item()
        y = root_quat_[0][2].item()
        z = root_quat_[0][3].item()
        
        # yaw (z-axis rotation)
        yaw1 = 2 * (w * z + x * y)
        yaw2 = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(yaw1, yaw2)

        # pitch (y-axis rotation)
        pitch1 = math.sqrt(1 + 2 * (w * y - x * z))
        pitch2 = math.sqrt(1 - 2 * (w * y - x * z))
        pitch = 2 * math.atan2(pitch1, pitch2) - 3.14 / 2

        # roll (x-axis rotation)
        roll1 = 2 * (w * x + y * z)
        roll2 = 1 - 2 * (x * x + y * y)
        roll = math.atan2(roll1, roll2)
    
        current_state = np.array([root_pos_[0][0].item(), root_pos_[0][1].item(), root_pos_[0][2].item(), yaw, pitch, roll, 0.0])  
        control, predicted_state = compute_mpc_control(mpc, current_state)
        vx_local = control[0] * np.cos(yaw) + control[1] * np.sin(yaw)
        vy_local = -control[0] * np.sin(yaw) + control[1] * np.cos(yaw)
        self._commands[0][0] = vx_local
        self._commands[0][1] = vy_local
        self._commands[0][2] = control[3]

        #print(vx_local, vy_local, control[3])

        print(self.f_x)
        self.count += 1
        self.t_list.append(self.count * (4/200))
        self.force_list.append(-self.f_x)

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
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                    #self._forces.squeeze(dim=1),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        self._P[:, 1:5][first_contact] = 1
        first_air = self._contact_sensor.compute_first_air(self.step_dt)[:, self._feet_ids]
        self._P[:, 1:5][first_air] = 0
        
        # reward machine
        self._extra_reward = torch.zeros(self.num_envs, 1, device=self.device)
        mask_delta = self._P[:, 0] > 0
        self._P[:, 0][mask_delta] -= 1

        maschera1 = (self._P == self._sequenza_target_1).all(dim=1)
        self._P[:, 5][maschera1] = 1
        self._extra_reward[maschera1] = 2
        self._P[:, 0][maschera1] = 10
        maschera2 = (self._P == self._sequenza_target_2).all(dim=1)
        self._P[:, 5][maschera2] = 0
        self._extra_reward[maschera2] = 2
        self._P[:, 0][maschera2] = 10
        #maschera3 = (self._P == self._sequenza_target_3).all(dim=1)
        #self._P[:, 5][maschera3] = 3
        #self._extra_reward[maschera3] = 2
        #self._P[:, 0][maschera3] = 5
        #maschera4 = (self._P == self._sequenza_target_4).all(dim=1)
        #self._P[:, 5][maschera4] = 0
        #self._extra_reward[maschera4] = 2
        #self._P[:, 0][maschera4] = 5
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

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        mask_extra = self._extra_reward > 0
        reward[mask_extra] += 0.075

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
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
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        numero = random.randint(1, 25)
        if numero == 10:
            self._commands[env_ids] *= 0

        #self._P[env_ids] *= 0
      
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




        # plot
        if (self.count > 10):
            plt.figure(figsize=(10, 8))

            # force plot
            plt.plot(self.t_list, self.force_list)
            plt.title('Interaction force', fontsize=18)
            plt.xlabel('Time [s]', fontsize=14)
            plt.ylabel('Force [N]', fontsize=14)
            plt.grid()

            # show plots
            plt.tight_layout()
            plt.show()