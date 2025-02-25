#!/usr/bin/env python3

import rospy
import tf
import moveit_commander
import cv2
import yaml
import os
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion

# Output directory
SAVE_PATH = os.path.expanduser("/home/chart-admin/koyo_ws/langsam_grasp_ws/src/demo_pkg_v2/src/calib_data")
os.makedirs(SAVE_PATH, exist_ok=True)

class KinovaCalibration:
    def __init__(self):
        rospy.init_node("kinova_realsense_calib", anonymous=True)

        # TF Listener
        self.tf_listener = tf.TransformListener()

        # Initialize planning group
        self.robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", 
            robot_description="my_gen3_lite/robot_description", 
            ns="/my_gen3_lite"
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            "gripper", 
            robot_description="my_gen3_lite/robot_description",
            ns="/my_gen3_lite"
        )

        # Set robot arm's speed and acceleration
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        ee_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % ee_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())

        print("============ Printing active joints in arm group")
        print(self.arm_group.get_active_joints())

        print("============ Printing joints in arm group")
        print(self.arm_group.get_joints())

        print("============ Printing active joints in gripper group")
        print(self.gripper_group.get_active_joints())

        print("============ Printing joints in gripper group")
        print(self.gripper_group.get_joints())

        # Image Subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # Camera Info Subscriber
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        
        self.camera_intrinsics = None  # Store camera parameters
        self.camera_intrinsics_saved = False  # Flag to ensure camera.xml is saved only once
        
        self.change_pos_count = 1

        # Image storage
        self.latest_image = None
        self.image_count = 1
        rospy.loginfo("Press SPACEBAR to capture image and pose. Press 'q' to quit.")
    
    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

    def camera_info_callback(self, msg):
        """Extracts camera intrinsic parameters from RealSense and saves them only once"""
        if self.camera_intrinsics_saved:
            return  # If camera.xml is already saved, skip

        self.camera_intrinsics = {
            "image_width": msg.width,
            "image_height": msg.height,
            "px": msg.K[0],  # Focal length in x (fx)
            "py": msg.K[4],  # Focal length in y (fy)
            "u0": msg.K[2],  # Principal point x (cx)
            "v0": msg.K[5],  # Principal point y (cy)
            "kud": msg.D[0] if len(msg.D) > 0 else 0,  # Distortion coefficient
            "kdu": 0,  # Reverse distortion (default 0)
        }

        self.save_camera_intrinsics()
        self.camera_intrinsics_saved = True  # Set flag to prevent re-saving
        self.init_target_joints()

    def save_camera_intrinsics(self):
        """Saves RealSense intrinsic parameters to camera.xml only if it hasn't been saved yet"""
        xml_filename = os.path.join(SAVE_PATH, "camera.xml")
        
        # Check if file already exists
        if os.path.exists(xml_filename):
            rospy.loginfo("Camera intrinsics already saved. Skipping.")
            return

        xml_content = f"""<?xml version="1.0"?>
<root>
<camera>
    <name>Camera</name>
    <image_width>{self.camera_intrinsics["image_width"]}</image_width>
    <image_height>{self.camera_intrinsics["image_height"]}</image_height>
    <model>
    <type>perspectiveProjWithDistortion</type>
    <px>{self.camera_intrinsics["px"]}</px>
    <py>{self.camera_intrinsics["py"]}</py>
    <u0>{self.camera_intrinsics["u0"]}</u0>
    <v0>{self.camera_intrinsics["v0"]}</v0>
    <kud>{self.camera_intrinsics["kud"]}</kud>
    <kdu>{self.camera_intrinsics["kdu"]}</kdu>
    </model>
</camera>
</root>"""

        with open(xml_filename, 'w') as f:
            f.write(xml_content)

        rospy.loginfo(f"Saved camera intrinsics to {xml_filename}")

    def image_callback(self, msg):
        """Stores the latest image from the camera"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def save_data(self):
        """Saves the latest image and corresponding end-effector pose"""
        if self.latest_image is None:
            rospy.logwarn("No image received yet.")
            return

        # Get transformation from base_link to end_effector_link
        try:
            (trans, rot) = self.tf_listener.lookupTransform("/base_link", "/tool_frame", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed")
            return

        # Convert quaternion to roll-pitch-yaw (Euler angles)
        rpy = euler_from_quaternion(rot)

        # Save image
        img_filename = os.path.join(SAVE_PATH, f"kinova_image_{self.image_count}.png")
        cv2.imwrite(img_filename, self.latest_image)

        # Save pose as YAML in the required format
        pose_filename = os.path.join(SAVE_PATH, f"kinova_pose_{self.image_count}.yaml")
        pose_data = {
            "rows": 6,
            "cols": 1,
            "data": [
                [float(trans[0])],  # X translation
                [float(trans[1])],  # Y translation
                [float(trans[2])],  # Z translation
                [float(rpy[0])],    # Roll
                [float(rpy[1])],    # Pitch
                [float(rpy[2])],    # Yaw
            ]
        }
        # Write YAML with custom formatting
        with open(pose_filename, 'w') as f:
            yaml.dump({"rows": 6, "cols": 1}, f, default_flow_style=False, sort_keys=False)
            f.write("data:\n")
            for value in pose_data["data"]:
                f.write(f"  - {value}\n")  # Forces correct `- [value]` formatting


        rospy.loginfo(f"Saved image {img_filename} and pose {pose_filename}")
        self.image_count += 1

    
    def go_to_position(self, joint_values):
        """
        Move the robot arm to the specified joint values.

        :param joint_values: List of target joint values [joint1, joint2, ..., jointN]
        """
        if len(joint_values) != 6:
            rospy.logerr("Error: The provided joint values do not match the robot's DOF.")
            return False

        self.arm_group.set_joint_value_target(joint_values)
        success = self.arm_group.go(wait=True)
        
        if success:
            rospy.loginfo("Successfully moved to the target position.")
        else:
            rospy.logerr("Failed to reach the target position.")

        return success
    
    def init_target_joints(self):
        self.target_joints1 = [-0.8401937680902645, -1.4499367502049205, 0.042248121216908026, -0.9900077646838001,-2.150676218342058, -2.2401064993274638]
        self.target_joints2 = [-0.740184612306737, -1.0079830367773992, 0.9459867771292263, -1.0432230495857633, -1.6526097007465665, -2.408891790262288]
        self.target_joints3 = [-0.5698733300672982, -0.7535552788755888, 1.3494141369632258, -1.1255775778709376, -1.2964896692522991, -2.3289823101739153]
        self.target_joints4 = [-0.3212954718159553, -0.5123186904851602, 1.650938833478651, -1.1764588683935555, -1.0697321550764123, -2.1010822998903844]
        self.target_joints5 = [-0.09516273523180185, -0.4217626913069772, 1.788605087075899, -1.1784967192596847, -0.9374401605589444, -1.9023439035432208]
        self.target_joints6 = [0.29876912373552045, -0.36123862710939925, 1.7783375358092077, -1.5251625288841701, -1.0172795995106476, -1.303073536099193]
        self.target_joints7 = [0.5446819566840202, -0.6728556388917895, 1.5091590612616668, -1.724666317728623, -1.0988532889642109, -0.8600492319134592]
        self.target_joints8 = [0.8726198848908514, -0.5629411217498212, 1.6409658278285222, -1.8848594520385236, -1.1639936765953314, -0.2801128813511884]
        self.target_joints9 = [1.028116800934507, -1.3627555087600864, 0.7700620839441179, -1.9266931853898157, -1.5629016520594652, -0.15134904264585725]
        self.target_joints10 = [0.3151949687069111, -0.15840482163791325, 2.620885538303102, -1.9113134300946077, -0.05506085553738771, -0.926510014813041]
        self.target_joints11 = [-0.5796370112557456, -0.587770305224848, 1.7374867752162642, -1.0961794752297713, -0.9743803354072149, -2.529965751320015]
        self.target_joints12 = [1.1688890993111145, -0.756218439965668, 1.997483735589648, -2.180085506259803, -0.846478295630634, 0.15090136526661485]
        self.target_joints13 = [0.47343041329949515, -1.3463003690167046, 0.21815071016496873, -1.6568931290438504, -2.0508481575126556, -0.9436245532423264]
        self.target_joints14 = [0.07769985533193384, -1.1328469423824181, 0.5398232855912796, -1.440780802377228, -1.7772232692091183, -1.3789145049902514]
        self.target_joints15 = [-0.027883829245348046, -1.3444041983205688, -0.030315295320590074, -1.2669839749030931, -2.1008343595928984, -0.6044709884207347]
    
    def change_pos(self):
        if self.change_pos_count == 1:
            self.go_to_position(self.target_joints1)
        elif self.change_pos_count == 2:
            self.go_to_position(self.target_joints2)
        elif self.change_pos_count == 3:
            self.go_to_position(self.target_joints3)
        elif self.change_pos_count == 4:
            self.go_to_position(self.target_joints4)
        elif self.change_pos_count == 5:
            self.go_to_position(self.target_joints5)
        elif self.change_pos_count == 6:
            self.go_to_position(self.target_joints6)
        elif self.change_pos_count == 7:
            self.go_to_position(self.target_joints7)
        elif self.change_pos_count == 8:
            self.go_to_position(self.target_joints8)
        elif self.change_pos_count == 9:
            self.go_to_position(self.target_joints9)
        elif self.change_pos_count == 10:
            self.go_to_position(self.target_joints10)
        elif self.change_pos_count == 11:
            self.go_to_position(self.target_joints11)
        elif self.change_pos_count == 12:
            self.go_to_position(self.target_joints12)
        elif self.change_pos_count == 13:
            self.go_to_position(self.target_joints13)
        elif self.change_pos_count == 14:
            self.go_to_position(self.target_joints14)
        elif self.change_pos_count == 15:
            self.go_to_position(self.target_joints15)


        self.change_pos_count += 1

    def run(self):
        """Main loop to display the image and wait for keyboard input"""
        self.go_sp()
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                # Show the latest camera image
                cv2.imshow("Camera Feed - Press SPACE to Capture, 'q' to Quit", self.latest_image)

                # Wait for keypress (10ms)
                key = cv2.waitKey(10) & 0xFF

                if key == 32:  # SPACEBAR key
                    self.change_pos()
                    self.save_data()
                elif key == ord('q'):  # 'q' key
                    rospy.loginfo("Quitting program.")
                    break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    calib = KinovaCalibration()
    calib.run()
