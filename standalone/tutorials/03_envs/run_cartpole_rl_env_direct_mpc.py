import argparse
from omni.isaac.lab.app import AppLauncher

# Lauching the simulator
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
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
from omni.isaac.lab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg
from omni.isaac.lab_tasks.direct.cartpole.cartpole_env import CartpoleEnv


def main():
    # Setup environment
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = CartpoleEnv(cfg=env_cfg)

    # Simulation loop
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
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
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # get current state as a numpy array
            x = obs["policy"][0][2].item()
            theta = obs["policy"][0][0].item()
            x_dot = obs["policy"][0][3].item()
            theta_dot = obs["policy"][0][1].item()
            current_state = np.array([-theta, x, -theta_dot, x_dot])
            if count % 10 == 0:
                print(current_state)
            # compute control action
            control, predicted_state = compute_mpc_control(mpc, current_state)
            # apply control action
            control = control.astype(np.float32)
            joint_effort = torch.tensor(control[0])
            joint_effort = joint_effort.view(1, 1)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_effort)
            count += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()