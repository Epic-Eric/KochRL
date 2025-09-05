from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

"""
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/folding \
  --tags clothes \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 50
"""

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

follower_arm = DynamixelMotorsBus(
            port="/dev/ttyACM0",
            motors={
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )

robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        #"koch_cam": OpenCVCamera(0, fps=30, width=640, height=480),
        "phone_cam": OpenCVCamera(1, fps=30, width=640, height=480),
    },
)

#koch_cam = OpenCVCamera(camera_index=0, fps=30, width=640, height=480)
phone_cam = OpenCVCamera(camera_index=1, fps=30, width=640, height=480)

if __name__ == "__main__":
    robot.connect()
    print("Both arms Connected!")   
    #koch_cam.connect()
    phone_cam.connect()

    try:
        observation, action = robot.teleop_step(record_data=True)
        print(observation["observation.images.koch_cam"].shape)
        print(observation["observation.images.phone_cam"].shape)
        print(observation["observation.images.koch_cam"].min().item())
        print(observation["observation.images.koch_cam"].max().item())
        # leader_pos = robot.leader_arms["main"].read("Present_Position")
        # follower_pos = robot.follower_arms["main"].read("Present_Position")

        # print(leader_pos)
        # print(follower_pos)
        #do = input("Action (read=r, manual=m, calib=c, current_control=o):")
        
    except KeyboardInterrupt:
        robot.disconnect()
        #koch_cam.disconnect()
        phone_cam.disconnect()