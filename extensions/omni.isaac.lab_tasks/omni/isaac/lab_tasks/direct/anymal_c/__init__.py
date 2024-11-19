"""
Anymal locomotion environment.
"""

import gymnasium as gym
from . import agents
#from .anymal_c_env import AnymalCEnv, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
from .anymal_c_env_stick import AnymalCEnv, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
#from .anymal_c_env_stick_training import AnymalCEnv, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
#from .anymal_c_env_stick_training2 import AnymalCEnv, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
#from .anymal_c_env_stick_training3 import AnymalCEnv, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg

# Register Gym environments.

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.anymal_c:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCFlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.anymal_c:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCRoughEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
