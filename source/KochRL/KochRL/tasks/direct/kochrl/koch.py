"""
Configuration file for Koch v1.1
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg  # Add this import

##
# Configuration
##

KOCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/asblab/ericxie/KochRL/source/KochRL/KochRL/tasks/direct/kochrl/follower.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=1, fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0
        },
    ),
    actuators={
        "koch_base": DCMotorCfg( #XL430-W250-T
            joint_names_expr=["joint_[1-2]"],
            saturation_effort=1.4, #Nm, peak torque
            effort_limit=1.0, #Nm, operating torque
            velocity_limit=5.969, 
            stiffness=10,
            damping=1,
        ),
        "koch_arm": DCMotorCfg( #XL330-M288-T
            joint_names_expr=["joint_[3-6]"],
            saturation_effort=0.52, #Nm, peak torque
            effort_limit=0.42, #Nm, operating torque
            velocity_limit=10.786, #rad/s
            stiffness=10,
            damping=1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Koch arm using implicit actuator models."""
