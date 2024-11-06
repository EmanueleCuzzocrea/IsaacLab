'''run_mpc.py'''

import argparse
from omni.isaac.lab.app import AppLauncher

# Launching the simulation
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Import mpc
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
taskFile = os.path.join(packageDir, 'config/franka/task.info')
libFolder = os.path.join(packageDir, 'auto_generated')
urdfDir = rospkg.RosPack().get_path('ocs2_robotic_assets')
urdf_ = os.path.join(urdfDir, 'resources/mobile_manipulator/franka/urdf/panda.urdf')

# Initialize MPC interface
mpc = mpc_interface(taskFile, libFolder, urdf_)

# State and input dimensions
stateDim = 7
inputDim = 7

# Set the goal
desiredTimeTraj = scalar_array()
desiredTimeTraj.push_back(0.0)
desiredInputTraj = vector_array()
desiredInputTraj.push_back(np.zeros(inputDim))
desiredStateTraj = vector_array()
desiredStateTraj.push_back(np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
targetTrajectories = TargetTrajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)
mpc.reset(targetTrajectories)


import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
# Pre-defined configs
from omni.isaac.lab_assets import FRANKA_PANDA_CFG, UR10_CFG  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Set goal
    ee_goals = [[0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    ik_commands = torch.zeros(scene.num_envs, 7, device=robot.device)
    ik_commands[:] = ee_goals[0]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            desiredTimeTraj = scalar_array()
            desiredTimeTraj.push_back(0.0)
            desiredInputTraj = vector_array()
            desiredInputTraj.push_back(np.zeros(inputDim))
            desiredStateTraj = vector_array()
            desiredStateTraj.push_back(np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
            targetTrajectories = TargetTrajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)
            mpc.reset(targetTrajectories)

            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
        joint_pos_ = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        joint1 = joint_pos_[0][0].item()
        joint2 = joint_pos_[0][1].item()
        joint3 = joint_pos_[0][2].item()
        joint4 = joint_pos_[0][3].item()
        joint5 = joint_pos_[0][4].item()
        joint6 = joint_pos_[0][5].item()
        joint7 = joint_pos_[0][6].item()
        current_state = np.array([joint1, joint2, joint3, joint4, joint5, joint6, joint7])
        print(current_state)
        # compute control action
        control, predicted_state = compute_mpc_control(mpc, current_state)
        control = control.astype(np.float32)
        # apply control action
        velocities = torch.tensor([control[0], control[1], control[2], control[3], control[4], control[5], control[6]], device=sim.device)
        robot.set_joint_velocity_target(velocities, joint_ids=robot_entity_cfg.joint_ids)
        # write data to sim
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # update marker positions
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    # Setup simulation config
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()