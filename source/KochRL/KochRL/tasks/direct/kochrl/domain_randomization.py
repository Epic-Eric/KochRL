from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg

@configclass
class EventCfg:
    """Configuration for domain randomization events."""
    
    # Joint stiffness and damping randomization for DYNAMIXEL servos
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint_[1-6]"),
            "stiffness_distribution_params": (0.9, 1.1),   # Moderate variation for servo precision
            "damping_distribution_params": (0.8, 1.2),     # Account for servo response variations
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    
    # Mass randomization to simulate payload variations
    robot_mass_randomization = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link_[4-6]"),  # End-effector links
            "mass_distribution_params": (0.9, 1.15),   # ±15% variation for payload simulation
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# Action noise for DYNAMIXEL servos (joint position commands)
action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(
        mean=0.0, 
        std=0.01,  # ~0.055° in radians - realistic servo jitter
        operation="add"
    ),
    bias_noise_cfg=GaussianNoiseCfg(
        mean=0.0, 
        std=0.008,  # ~0.46° systematic offset per episode
        operation="add"
    ),
)

# Observation noise for sensor readings
observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(
        mean=0.0, 
        std=0.01,  # ~0.57° for joint positions, scaled for other observations
        operation="add"
    ),
    bias_noise_cfg=GaussianNoiseCfg(
        mean=0.0, 
        std=0.001,  # Small systematic sensor offset
        operation="add"
    ),
)