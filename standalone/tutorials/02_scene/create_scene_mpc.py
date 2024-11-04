"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.
# Usage
./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# Launching the simulator
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Import mpc
import numpy as np
import os
import rospkg
from ocs2_cartpole import mpc_interface
from ocs2_cartpole import (
    scalar_array,
    vector_array,
    TargetTrajectories,
)
from CartPolePyBindingMPC import compute_mpc_control

# Path of the package
packageDir = rospkg.RosPack().get_path('ocs2_cartpole')
taskFile = os.path.join(packageDir, 'config/mpc/task.info')
libFolder = os.path.join(packageDir, 'auto_generated')

# Initialize MPC interface
mpc = mpc_interface(taskFile, libFolder)

# State and input dimensions
stateDim = 4
inputDim = 1

# Set the goal
desiredTimeTraj = scalar_array()
desiredTimeTraj.push_back(0.0)
desiredInputTraj = vector_array()
desiredInputTraj.push_back(np.zeros(inputDim))
desiredStateTraj = vector_array()
desiredStateTraj.push_back(np.zeros(stateDim))
targetTrajectories = TargetTrajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)
mpc.reset(targetTrajectories)


import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["cartpole"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        # Reset
        if count % 800 == 0:
            # reset mpc
            desiredTimeTraj = scalar_array()
            desiredTimeTraj.push_back(0.0)
            desiredInputTraj = vector_array()
            desiredInputTraj.push_back(np.zeros(inputDim))
            desiredStateTraj = vector_array()
            desiredStateTraj.push_back(np.zeros(stateDim))
            targetTrajectories = TargetTrajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)
            mpc.reset(targetTrajectories)
            
            # reset simulation
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            #joint_pos += torch.rand_like(joint_pos) * 0.1
            joint_pos[0][1] += 3.14
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # get current state as a numpy array
        joint_pos_, joint_vel_ = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        x = joint_pos_[0][0].item()
        theta = joint_pos_[0][1].item()
        x_dot = joint_vel_[0][0].item()
        theta_dot = joint_vel_[0][1].item()
        current_state = np.array([-theta, x, -theta_dot, x_dot])
        print(current_state)
        # compute control action
        control, predicted_state = compute_mpc_control(mpc, current_state)
        # apply control action
        efforts = torch.tensor([control[0], 0.0])
        robot.set_joint_effort_target(efforts)
        # write data to sim
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    # Set simulation config
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()