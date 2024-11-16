"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

# Launch the simulator
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
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


import math
import torch
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=0.8)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.2),
            "operation": "add",
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (1.0, 2.0),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # update the control every 4 simulation step
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    # create environment
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
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
            x = obs["policy"][0][0].item()
            theta = obs["policy"][0][1].item()
            x_dot = obs["policy"][0][2].item()
            theta_dot = obs["policy"][0][3].item()
            current_state = np.array([-theta, x, -theta_dot, x_dot])
            print(current_state)
            # compute control action
            control, predicted_state = compute_mpc_control(mpc, current_state)
            # apply control action
            joint_effort = torch.tensor(control[0])
            joint_effort = joint_effort.view(1, 1)
            # step the environment
            obs, _ = env.step(joint_effort)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()