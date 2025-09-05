from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
leader_arm = DynamixelMotorsBus(
            port="/dev/ttyACM1",
            motors={
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )

"""
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/folding \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_koch_folding \
  hydra.job.name=act_koch_folding \
  device=cuda \
  wandb.enable=true
"""