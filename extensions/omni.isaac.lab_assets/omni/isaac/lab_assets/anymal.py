"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.actuators import ImplicitActuatorCfg

from .velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

##
# Configuration - Actuators.
##

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)
"""Configuration for ANYdrive 3.x with DC actuator model."""


ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##

ANYMAL_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-B robot using actuator-net."""


ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""

ANYMAL_KINOVA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/emanuele/isaac/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Loco-manipulation-kinova2/anymal_kinova.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*F_HFE": 0.4,     # both front HFE
            ".*H_HFE": -0.4,    # both hind HFE
            ".*F_KFE": -0.8,    # both front KFE
            ".*H_KFE": 0.8,     # both hind KFE
            "j2n6s300_joint_1": 0.0,
            "j2n6s300_joint_2": 2.76,
            "j2n6s300_joint_3": 2.76,
            "j2n6s300_joint_4": 2.5,
            "j2n6s300_joint_5": 2.0,
            "j2n6s300_joint_6": 0.0,
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG,
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_[1-6]"],
            velocity_limit=100.0,
            effort_limit={
                ".*_joint_[1-6]": 100.0,
            },
            stiffness={
                ".*_joint_[1-6]": 0.0,
            },
            damping={
                ".*_joint_[1-6]": 100.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot with a kinova arm."""










ANYMAL_STICK_F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/emanuele/isaac/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Loco-manipulation-stick-front/anymal_d_stick_front.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
            #collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00001, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        #rot=(0.7, 0.0, 0.0, 0.7),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*F_HFE": 0.4,     # both front HFE
            ".*H_HFE": -0.4,    # both hind HFE
            ".*F_KFE": -0.8,    # both front KFE
            ".*H_KFE": 0.8,     # both hind KFE
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)




ANYMAL_STICK_L_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/emanuele/isaac/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Loco-manipulation-stick-lateral/anymal_d_stick_lateral.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        #collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        #rot=(0.7, 0.0, 0.0, 0.7),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*F_HFE": 0.4,     # both front HFE
            ".*H_HFE": -0.4,    # both hind HFE
            ".*F_KFE": -0.8,    # both front KFE
            ".*H_KFE": 0.8,     # both hind KFE
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)




ANYMAL_STICK_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/emanuele/isaac/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Loco-manipulation-stick-diagonal/anymal_d_stick_diagonal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        #collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        #rot=(0.92, 0.0, 0.0, 0.38),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*F_HFE": 0.4,     # both front HFE
            ".*H_HFE": -0.4,    # both hind HFE
            ".*F_KFE": -0.8,    # both front KFE
            ".*H_KFE": 0.8,     # both hind KFE
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
















##
# Configuration - Sensors.
##

ANYMAL_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
    offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
)
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
