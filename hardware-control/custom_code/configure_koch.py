from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
#from camera import ZEDCamera

import subprocess
import cv2
import time
import numpy as np
try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None
try:
    from roboticstoolbox import DHRobot, RevoluteDH
except Exception:
    DHRobot = None
    RevoluteDH = None
try:
    from pynput import keyboard
except Exception:
    keyboard = None
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.draw import disk

import threading

class KochLeader:
    def __init__(self, port):

        # sudo enable robot USB port
        command = ["sudo", "chmod", "666", port]
        subprocess.run(command, check=True, text=True, capture_output=True)

        leader_arm = DynamixelMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )

        self.robot = ManipulatorRobot(
            robot_type="koch",
            leader_arms={"main": leader_arm},
            calibration_dir=".cache/calibration/koch",
        )
        self.robot.connect()
        print("Leader Connected!")

        torque_mode = TorqueMode.DISABLED.value
        self.robot.leader_arms["main"].write("Torque_Enable", torque_mode)

    def get_arm_joint_angles(self):
        """
        Reads current joint angles of each revolute joint.
        """
        angles = self.robot.leader_arms["main"].read("Present_Position")
        angles = np.array(angles)
        return angles
    
    def exit(self):
        self.robot.disconnect()

class KochFollower:

    def __init__(self, port, torque):

        # sudo enable robot USB port
        command = ["sudo", "chmod", "666", port]
        subprocess.run(command, check=True, text=True, capture_output=True)

        follower_arm = DynamixelMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )

        self.robot = ManipulatorRobot(
            robot_type="koch",
            follower_arms={"main": follower_arm},
            calibration_dir=".cache/calibration/koch",
        )
        self.robot.connect()
        print("Follower Connected!")

        # DH parameters and limits
        self.d1, self.a2, self.a3, self.d5 = 5.5, 10.68, 10, 10.5
        self.A = [0, self.a2, self.a3, 0, 0]
        self.ALPHA = [-np.pi/2, 0, 0, np.pi/2, 0]
        self.D = [self.d1, 0, 0, 0, self.d5]
        self.OFFSET = [0, 0, 0, np.pi/2, 0]
        QLIM = [[-np.pi/2, np.pi/2], [0, np.pi/2], [-np.pi/2, np.pi/2],\
                [-np.pi/2, np.pi/2], [-np.pi, np.pi]]
        self.LLIM = [[8, 18], [-15, 15], [1, 20], [], [-np.pi/2, np.pi/2], [0, np.pi/2]]
        
        # Initial EE angles
        self.q4, self.q5 = 0, np.pi / 2
        
        # For computing forward kinematics
        self.robot_dh = DHRobot([RevoluteDH(a=self.A[i], 
                                            alpha=self.ALPHA[i], 
                                            d=self.D[i], 
                                            offset=self.OFFSET[i], 
                                            qlim=QLIM[i]) for i in range(len(self.A))])

        # Initialize with torque enabled/disabled
        torque_mode = TorqueMode.ENABLED.value if torque else TorqueMode.DISABLED.value
        self.robot.follower_arms["main"].write("Torque_Enable", torque_mode)

        # Set to home position
        # self.home_position = [10, 0, 5.5]
        # self.set_to_home()

        # time.sleep(1)
        # self.curr_position = self.get_ee_pos()


    def get_arm_joint_angles(self):
        """
        Reads current joint angles of each revolute joint.
        """
        angles = self.robot.follower_arms["main"].read("Present_Position")
        angles = np.array(angles) /  360 * 2 * np.pi
        return angles


    def get_ee_pose(self):
        """
        Reads joint angles and computes ee pose.
        """

        def computeDHMatrix(a, alpha, d, theta):

            ca = np.cos(alpha)
            sa = np.sin(alpha)
            ct = np.cos(theta)
            st = np.sin(theta)
            
            # Construct the DH matrix
            T = np.array([
                [ ct,           -st * ca,        st * sa,         a * ct ],
                [ st,            ct * ca,       -ct * sa,         a * st ],
                [ 0,             sa,             ca,              d      ],
                [ 0,             0,              0,               1      ]
            ], dtype=float)
            
            return T

        # Follower arms outputs 6 positions because there are 6 motors
        robot_angles_full = self.get_arm_joint_angles()

        # However, we only need the first five for DH - the last one is gripper open/close
        robot_angles = np.array(robot_angles_full)[:5]
        robot_angles[4] = np.abs(robot_angles[4])
        robot_angles = np.round(robot_angles, decimals=2)

        # Between self.robot and self.robot_dh, the difference is that first and second angle is flipped
        # Note that first angle is flipped to ensure right-handed axis
        robot_angles *= np.array([-1, -1, 1, 1, 1])
        
        # Forward kinematics
        # T = np.eye(4)
        # for i in range(5):
        #     Ti = computeDHMatrix(self.A[i], self.ALPHA[i], self.D[i], robot_angles[i] + self.OFFSET[i])
        #     T = T @ Ti

        T = np.array(self.robot_dh.fkine(robot_angles))
        # T_fkine = np.array(self.robot_dh.fkine(robot_angles))
        # print(T, T_fkine)
        # assert np.array_equal(T, T_fkine)

        ee_matrix, ee_pos = T[:3, :3], T[:3, 3]
        # print(robot_angles)
        quat_xyzw = R.from_matrix(ee_matrix).as_quat()

        # Verify correctness of inverse kinematics
        Q_from_inv = self.inv_kin(ee_pos)
        diff = np.linalg.norm(Q_from_inv[:4] - robot_angles[:4])
        
        ee_pose_array = np.hstack((ee_pos, quat_xyzw))

        print("\nEnd-Effector Pose (Position + Quaternion):")
        print("[" + ", ".join(f"{val:>12.8f}" for val in ee_pose_array) + "]")

        print("\nHomogeneous Transformation Matrix (4x4):")
        for row in T:
            print("[" + "  ".join(f"{val:>12.8f}" for val in row) + "]")
        
        return np.hstack((ee_pos, quat_xyzw)), T

    def get_ee_pos(self):
        return self.get_ee_pose()[0][:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[0][3:]
    
    def get_transformation(self):
        return self.get_ee_pose()[1]
    

    def inv_kin(self, target_position):

        px, py, pz = target_position

        # Analytic Solver
        r1 = np.sqrt(self.d5**2 + self.a3**2)
        r2 = np.sqrt(px**2 + py**2 + (pz-self.d1)**2)

        q1 = np.arctan2(py, px)

        phi_1 = np.arctan2(pz-self.d1, np.sqrt(px**2 + py**2))
        D2 = (self.a2**2 + r2**2 - r1**2) / (2*self.a2*r2)
        phi_2 = np.arctan2(np.sqrt(1-D2**2), D2)
        q2 = phi_1 + phi_2

        phi_4 = np.arctan2(self.d5, self.a3)
        D = (r2**2 - self.a2**2 - r1**2) / (2*self.a2*r1)
        q3 = np.arctan2(np.sqrt(1-D**2), D) - phi_4
        phi_3 = np.pi - q3 - phi_4

        Q = [-q1, -q2, q3, np.pi/2, -q1, self.q5]

        return np.array(Q)
    
    
    def set_ee_pose(self, pos, axis, steps=50):

        def verify_inv(q):
            limits = self.LLIM[axis]
            if np.isnan(q).any():
                return False
            if 0 <= axis <= 2 and not limits[0] <= pos[axis] <= limits[1]:
                return False
            elif 4 <= axis <= 5 and not limits[0] <= q[axis] <= limits[1]:
                return False            
            return True

        q = self.inv_kin(pos)

        if verify_inv(q):
            q[1] *= -1
            curr_q = self.get_arm_joint_angles()
            interp = np.linspace(curr_q, q, num=steps) / (2 * np.pi) * 360
            for q_int in interp[1:]:
                time.sleep(0.005)
                self.robot.follower_arms["main"].write("Goal_Position", q_int)
                self.curr_position = self.get_ee_pos()
            return True
        else:
            return False

    def set_gripper_open(self):
        self.q5 = 0
        self.set_ee_pose(self.get_ee_pos(), axis=5, steps=50)

    def set_gripper_close(self):
        self.q5 = np.pi / 2
        self.set_ee_pose(self.get_ee_pos(), axis=5, steps=50)

    def set_to_home(self):
        self.q4 = 0
        self.q5 = np.pi / 2
        self.set_ee_pose(self.home_position, axis=0, steps=50)


    def manual_control(self):

        self.curr_position = self.get_ee_pos()
        
        # increment
        m = 0.05
        # xyz = position, eeo = end-effector orientation, eeg = grip open/close
        control_dict = {"1": {"ctrl": "xyz", "axis": 0, "direc": 1},
                        "2": {"ctrl": "xyz", "axis": 0, "direc": -1},
                        "3": {"ctrl": "xyz", "axis": 1, "direc": 1},
                        "4": {"ctrl": "xyz", "axis": 1, "direc": -1},
                        "5": {"ctrl": "xyz", "axis": 2, "direc": 1},
                        "6": {"ctrl": "xyz", "axis": 2, "direc": -1},
                        "7": {"ctrl": "eeo", "axis": 4, "direc": 1},
                        "8": {"ctrl": "eeo", "axis": 4, "direc": -1},
                        "9": {"ctrl": "eeg", "axis": 5, "direc": 1},
                        "0": {"ctrl": "eeg", "axis": 5, "direc": -1},
                       }

        def on_press(key):

            if hasattr(key, 'char'):

                # Exit
                if key.char == "x":
                    listener.stop()
                    return
                
                # Return to start position
                elif key.char == "h":
                    self.set_to_home()
                    self.curr_position = self.get_ee_pos()
                
                elif key.char in control_dict:

                    # Get control parameters
                    control = control_dict[key.char]
                    ctrl, axis, direc = control["ctrl"], control["axis"], control["direc"]

                    # Set position
                    if ctrl == "xyz":
                        self.curr_position[axis] += direc * m
                    elif ctrl == "eeo":
                        self.q4 += direc * m
                    elif ctrl == "eeg":
                        self.set_gripper_open() if direc == 1 else self.set_gripper_close()

                    # Revert if command has reached limit
                    if not self.set_ee_pose(self.curr_position, axis, steps=2):
                        if ctrl == "xyz":
                            self.curr_position[axis] -= direc * m
                        elif ctrl == "eeo":
                            self.q4 -= direc * m
                        print("Limit reached!")

                    print("Position:", self.curr_position)

                    #rgb, point = self.return_estimated_ee(cam_node)
                    #cv2.imwrite("x.png", rgb)


        print("Axis Controls: 2<-x->1 , 4<-y->3, 6<-z->5")
        print("Gripper Controls: 7<-z->8 , 9<-g->0")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        return
    

    def get_point_to_world_conversion(self, camera):
        
        # Get limits of end-effector
        x_limits, y_limits, z_limits = self.LLIM[0], self.LLIM[1], self.LLIM[2]
        x_interp = np.linspace(x_limits[0], x_limits[1], num=100)
        y_interp = np.linspace(y_limits[0], y_limits[1], num=100)
        z_interp = np.linspace(z_limits[0], z_limits[1], num=100)

        # Iterate through points
        print("Getting point to world dict...")
        t = time.time()
        self.point_to_world = {}
        for x in x_interp:
            for y in y_interp:
                for z in z_interp:
                    point = self.convert_world_to_point(camera, [x, y, z])
                    self.point_to_world[tuple(point)] = [x,y,z]
        print("Time:", time.time() - t)
        return self.point_to_world

    def convert_world_to_point(self, cam_node, world_coord):
        T = np.hstack((cam_node.R, cam_node.t))   # The saved t is converted from m to cm
        P = np.array([[world_coord[0], world_coord[1], world_coord[2], 1]])
        pc = T @ P.T
        x_star = cam_node.K @ pc
        x_star = x_star / x_star[2]
        point = [int(np.rint(x_star[1])[0]), int(np.rint(x_star[0])[0])]
        return point

    def return_estimated_ee(self, cam_node):
        """
        Get estimated pixel coordinate position of the end-effector.
        """

        # Compute pixel coordinate
        new_world = self.curr_position
        point = self.convert_world_to_point(cam_node, new_world)
        rgb = cam_node.capture_image("rgb")
        rr, cc = disk(point, 10, shape=rgb.shape)
        rgb[rr, cc] = (255, 255, 0)

        return rgb, point
    
    def find_closest_point_to_world(self, ref_point):

        # Convert dictionary keys (pixel points) to a NumPy array for fast distance computation
        pixel_points = np.array(list(self.point_to_world.keys()))
        
        # Compute Euclidean distances from ref_point to all pixel points
        distances = np.linalg.norm(pixel_points - np.array(ref_point), axis=1)
        
        # Find the index of the closest pixel point
        closest_index = np.argmin(distances)
        
        # Retrieve the closest pixel point and its corresponding world point
        closest_pixel_point = tuple(pixel_points[closest_index])  # Convert back to tuple
        corresponding_world_point = self.point_to_world[closest_pixel_point]

        return corresponding_world_point


    def camera_extrinsics(self, cam_node):

        # Pick up object for ee-tracking
        print("Pick up the object to track the end effector...")
        self.manual_control(cam_node)
        
        # Hardcoded poses - we will capture all intermediate poses automatically
        move_poses = [[9, -13, 15], [16.74, -14.48, 5.1],\
                      [12.83, -1.57, 7], [9, 13, 15],
                      [17.46, 15, 8]]
        calibration_poses, calibration_coords = [], []

        # Write video
        coords, frame = cam_node.detect_end_effector()
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for AVI
        out = cv2.VideoWriter("calibration.mp4", fourcc, 20, (width, height))

        for final_position in move_poses:
            
            # Generate positions from current position to end position
            curr_position = self.get_ee_pos()
            lin_x = np.linspace(curr_position[0], final_position[0], num=100)
            lin_y = np.linspace(curr_position[1], final_position[1], num=100)
            lin_z = np.linspace(curr_position[2], final_position[2], num=100)
            grad_poses = np.vstack((lin_x, lin_y, lin_z)).T

            # Move end effector to goal
            print("Sending to goal...")
            
            for i, g in enumerate(grad_poses):
                time.sleep(0.02)
                self.set_ee_pose(g, axis=0, steps=2)

                # Get end-effector coordinates from the blue object
                coords, frame = cam_node.detect_end_effector()
                calibration_coords += [coords]
                calibration_poses += [grad_poses[i]]

                # Add frame and save video
                out.write(frame)       

            print("Reached!")
            time.sleep(1)   # wait before capturing picture
        
        # Save video
        out.release()

        calibration_poses = np.array(calibration_poses).reshape(-1,3).astype(np.float32)
        calibration_coords = np.array(calibration_coords).reshape(-1,2).astype(np.float32)

        # Estimate the rotation vector (rvec) and translation vector (tvec)
        _, rvec, t = cv2.solvePnP(calibration_poses, calibration_coords, cam_node.K, distCoeffs=cam_node.D)
        R, _ = cv2.Rodrigues(rvec)

        # Save output
        print("R:", R, "t:", t)
        np.save("camera_extrinsics.npy", np.array([R.reshape(-1), t], dtype=object))

        return [R, t]


    def exit(self):
        input("Press return to deactivate robot...")
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.disconnect()
    
    def test_current_control(self):
        # safety_factor = 2.0
        # max_current_value = self.robot.follower_arms["main"].read("Current_Limit", motor_names=["gripper"]) / safety_factor
        # self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        # self.robot.follower_arms["main"].write("Operating_Mode", 0, motor_names=["gripper"])
        # self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
        # self.robot.follower_arms["main"].write("Goal_Current", 50, motor_names=["gripper"]) # 1750mA = 100% current
        # now=time.time()
        # while time.time() - now < 5:
        #     print("Current current: ", self.robot.follower_arms["main"].read("Present_Current", motor_names=["gripper"]))
        
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.follower_arms["main"].write("Operating_Mode", 0, motor_names=["shoulder_pan"])
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value, motor_names=["shoulder_pan"])
        self.robot.follower_arms["main"].write("Goal_Current", 200, motor_names=["shoulder_pan"]) # 0.113% * 885 = 100% duty cycle
    
    def print_current(self):
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.follower_arms["main"].write("Operating_Mode", 3, motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value, motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
        # self.robot.follower_arms["main"].write("Goal_Position", [70, 30, 0], motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
        # while True:
        #     print("Goal Position:", self.robot.follower_arms["main"].read("Goal_Position", motor_names=["gripper"])[0])
        #     print("Present Position:", self.robot.follower_arms["main"].read("Present_Position", motor_names=["gripper"])[0])
        #     time.sleep(0.01)
        plt.ion()
        current = [0]
        t = [0]
        graph = plt.plot(t, current)[0]
        second = 0
        plt.pause(0.1)
        print(type(graph))
        print(graph)
        angle = 0
        dir = 1
        while True:
            if angle > 90:
                dir = 0
            elif angle < -90:
                dir = 1
            if dir:
                angle += 4
            else:
                angle -= 4
            self.robot.follower_arms["main"].write("Goal_Position", angle, motor_names=["wrist_flex"])
            unsigned = self.robot.follower_arms["main"].read("Present_Current", motor_names=["wrist_flex"])[0]
            if unsigned > 32767:
                unsigned -= 65536
            current.append(unsigned)
            second += 0.1
            t.append(second)
            graph.remove()
            graph = plt.plot(t,current,color = 'g')[0]
            plt.xlim(t[0], t[-1])
            plt.xlabel("Time")
            plt.ylabel("Current (mA)")
            plt.title("Current Over Time")
            plt.pause(0.1)
    
    def gravity_compensation(self):
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.follower_arms["main"].write("Operating_Mode", 3, motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value, motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
        while True:
            angles = self.robot.follower_arms["main"].read("Present_Position", motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
            self.robot.follower_arms["main"].write("Goal_Position", angles, motor_names=["shoulder_lift", "elbow_flex", "wrist_flex"])

def moving_window_average(data, window_size=5):
    """
    Computes the moving window average of the input data.
    """
    if len(data) < window_size:
        return data
    avg = np.mean(data[-window_size:])
    data[-(window_size//2+1)] = avg
    return data

def initialization(koch_follower:KochFollower, koch_leader:KochLeader):
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    koch_follower.robot.follower_arms["main"].write("Operating_Mode", 3) # Position Control Mode
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    leader_angles = koch_leader.get_arm_joint_angles()
    koch_follower.robot.follower_arms["main"].write("Goal_Position", leader_angles)
    time.sleep(2)
    print("Follower initialized to leader's position.\n")

def plot(motor_names, current_data, t, axes, graph_lines):
    # Update data in the plot
    for i, motor_name in enumerate(motor_names):
        graph_lines[i].remove()
        graph_lines[i] = axes[i].plot(t, current_data[motor_name], color='g')[0]

        axes[i].set_xlim(t[0], t[-1])
        y_data = current_data[motor_name]
        if len(y_data) > 1:
            y_min, y_max = min(y_data), max(y_data)
            y_range = y_max - y_min
            axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    plt.pause(0.001)

def follow(koch_follower:KochFollower, koch_leader:KochLeader, configure_mode=False, bilateral_control=False):
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    koch_follower.robot.follower_arms["main"].write("Operating_Mode", 3) # Position Control Mode
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    total_angles = []
    # Plotting
    plt.ion()
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    motor_titles = ["Shoulder Pan Current", "Shoulder Lift Current", "Elbow Flex Current",
                   "Wrist Flex Current", "Wrist Roll Current", "Gripper Current"]
    
    current_data = {name: [0] for name in motor_names}
    t = [0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    #display
    graph_lines = []
    for i, (motor_name, title) in enumerate(zip(motor_names, motor_titles)):
        axes[i].set_title(title)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Current (mA)")
        line = axes[i].plot(t, current_data[motor_name], color='g')[0]
        graph_lines.append(line)
    
    plt.tight_layout()
    plt.pause(0.001)

    t1 = threading.Thread(target=plot, args=(motor_names, current_data, t, axes, graph_lines))
    #t1.start()

    #bilateral control initilization
    if bilateral_control:
        current_threshold = {}
        with open("current_thresholds.txt", "r") as f:
            lines = f.readlines()
        for i, motor_name in enumerate(motor_names):
            min_val, max_val = map(float, lines[i].strip().split())
            current_threshold[motor_name] = (min_val, max_val)

    while True:
        try:
            start = time.time()
            leader_angles = koch_leader.get_arm_joint_angles()
            # print("leader: ", leader_angles[4])
            # print("follower: ", koch_follower.get_arm_joint_angles()[4] / np.pi * 180)
            total_angles.append(leader_angles)
            koch_follower.robot.follower_arms["main"].write("Goal_Position", leader_angles)
            follower_angles = koch_follower.get_arm_joint_angles()

            # Get new data
            for i, motor_name in enumerate(motor_names):
                if bilateral_control:
                    if current_data[motor_name][-1] < current_threshold[motor_name][0] or current_data[motor_name][-1] > current_threshold[motor_name][1]:
                        koch_leader.robot.leader_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value, motor_names=[motor_name])
                        koch_leader.robot.leader_arms["main"].write("Operating_Mode", 1, motor_names=[motor_name]) # Current control
                        koch_leader.robot.leader_arms["main"].write("Goal_Current", -(current_data[motor_name][-1]), motor_names=[motor_name])
                        #koch_leader.robot.leader_arms["main"].write("Goal_Position", [follower_angles[i]], motor_names=[motor_name])
                    else:
                        koch_leader.robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value, motor_names=[motor_name])

                unsigned = koch_follower.robot.follower_arms["main"].read("Present_Current", motor_names=[motor_name])[0]
                if unsigned > 32767:
                    unsigned -= 65536
                current_data[motor_name].append(unsigned)
                current_data[motor_name] = moving_window_average(current_data[motor_name], window_size=5)
                        
            t.append(t[-1] + time.time() - start)
            # Update data in the plot
            #plot(motor_names, current_data, t, axes, graph_lines)

        except KeyboardInterrupt:
            break
    # current_threshold[motor_name] = (min, max)
    if configure_mode:
        current_threshold = {}
        for i, motor_name in enumerate(motor_names):
            current_threshold[motor_name] = (min(current_data[motor_name]), max(current_data[motor_name]))
        with open("current_thresholds.txt", "w") as f:
            for (min_val, max_val) in current_threshold.values():
                f.write(f"{min_val} {max_val}\n")
        print("Current thresholds saved to current_thresholds.txt")

    with open("blind_follow_angles.txt", "w") as f:
        for angles in total_angles:
            f.write(" ".join([str(angle) for angle in angles]) + "\n")

def copy_from_movement_file(koch_follower:KochFollower, filename="blind_follow_angles.txt"):
    """
    Copy angles from a file and move the robot accordingly.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    angles_list = [list(map(float, line.strip().split())) for line in lines]
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    koch_follower.robot.follower_arms["main"].write("Operating_Mode", 3)  # Position Control Mode
    koch_follower.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    for angles in angles_list:
        angles = np.array(angles)
        koch_follower.robot.follower_arms["main"].write("Goal_Position", angles)
        time.sleep(0.005)

if __name__ == "__main__":

    FOLLOWER_PORT = "/dev/ttyACM0"
    LEADER_PORT = "/dev/ttyACM1"
    ENABLE_TORQUE = False

    koch_follower = KochFollower(port=FOLLOWER_PORT, torque=ENABLE_TORQUE)
    koch_leader = KochLeader(port=LEADER_PORT)

    try:
        do = input("Action (read=r, manual=m, calib=c, current_control=o):")
        if do == "r":
            while True:
                koch_follower.get_ee_pose()
                time.sleep(0.1)
        elif do == "m":
            koch_follower.manual_control()
        elif do == "c":
            pass
            #koch_robot.camera_extrinsics(cam_node)
        elif do == "o":
            koch_follower.print_current()
        elif do == "f":
            #initialization(koch_follower, koch_leader)
            follow(koch_follower, koch_leader, configure_mode=False, bilateral_control=True)
        elif do == "x":
            copy_from_movement_file(koch_follower)
        elif do == "y":
            print(koch_follower.robot.follower_arms["main"].read("Present_Position"))
    except KeyboardInterrupt:
        koch_follower.exit()
        koch_leader.exit()

    koch_follower.exit()
    koch_leader.exit()
    

    
