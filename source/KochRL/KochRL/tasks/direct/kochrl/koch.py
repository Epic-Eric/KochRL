"""
Configuration file for Koch v1.1
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

KOCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="follower.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 3.14,
            "joint_2": 3.14,
            "joint_3": 3.05,
            "joint_4": 3.14,
            "joint_5": 6.28,
            "joint_6": 4.6
        },
    ),
    actuators={
        "koch_base": ImplicitActuatorCfg( #XL430-W250-T
            joint_names_expr=["joint_[1-2]"],
            effort_limit=0.28,
            velocity_limit=5.969, 
            stiffness=10,
            damping=1,
        ),
        "koch_arm": ImplicitActuatorCfg( #XL330-M288-T
            joint_names_expr=["joint_[3-6]"],
            effort_limit=0.26, #Nm
            velocity_limit=10.786, #rad/s
            stiffness=10,
            damping=1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Koch arm using implicit actuator models."""
