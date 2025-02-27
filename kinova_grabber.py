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
        self.arm_group.set_joint_value_target([0.02754534147079857, -0.3292162455300689, 
                                               0.6239125970105316, -1.5710093796821027, 
                                               -2.1819422621718063, -1.6193681240201974])
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

    # def image_callback(self, msg):
    #     """Stores the latest image from the camera"""
    #     try:
    #         self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #     except Exception as e:
    #         rospy.logerr(f"Error processing image: {e}")
    
    def image_callback(self, msg):
        """Enhances the contrast of the chessboard by making white squares brighter without affecting black squares."""
        try:
            # Convert ROS image to OpenCV format
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert to grayscale
            gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)

            # Apply a gentle brightness boost ONLY to white areas
            white_boosted = cv2.addWeighted(gray, 1.5, np.zeros_like(gray), 0, 50)

            # Ensure the brightness doesn't exceed the max value (255)
            white_boosted = np.clip(white_boosted, 0, 255).astype(np.uint8)

            # Replace stored image with improved version
            self.latest_image = cv2.cvtColor(white_boosted, cv2.COLOR_GRAY2BGR)

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
        img_filename = os.path.join(SAVE_PATH, f"image_{self.image_count}.png")
        cv2.imwrite(img_filename, self.latest_image)

        # Save pose as YAML in the required format
        pose_filename = os.path.join(SAVE_PATH, f"pose_fPe_{self.image_count}.yaml")
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
        self.target_joints1 = [0.3353550981588109, -0.4843043663463984, 1.793778011177269, -1.573439780492909, -0.8595203281209702, -1.3098235841981074]
        self.target_joints2 = [0.45487430577215004, -0.1667985727616239, 2.269328567020576, -1.649195528229085, -0.5257708497833677, -1.1408944825644198]
        self.target_joints3 = [0.5693257841471774, 0.0028503813147118086, 2.559310058107162, -1.9040304834615682, -0.32133275607121714, -0.7868272156383851]
        self.target_joints4 = [0.6217583660047664, 0.06014216689725944, 2.6214527916152894, -1.946093781298825, -0.32806416204250066, -0.6958509696401878]
        self.target_joints5 = [0.22702835765874785, -0.8824362966688835, 1.0448640894494703, -1.502655089247475, -1.450627574191687, -1.4548369666106664]
        self.target_joints6 = [0.20921473944331687, -1.1115565673638885, 0.6088084788881472, -1.4988638131196383, -1.7372870381345082, -1.4580066609400788]
        self.target_joints7 = [0.20013043064894756, -1.3459669412482276, 0.19136729908204184, -1.4793082212351871, -2.048214823826785, -1.44669941158382]
        self.target_joints8 = [0.19829737687064602, -1.408137904263909, 0.04275438814013209, -1.4760788720973563, -2.137722869116022, -1.4395237903427107]
        self.target_joints9 = [-0.7095550639775174, -1.3705990508025883, 0.6435443553021596, -1.037430674214841, -1.664845328058827, -2.4585450972062697]
        self.target_joints10 = [-0.5999137871633913, -1.036096430508712, 1.103433393406929, -1.0597085493655722, -1.415565993176358, -2.436751384741716]
        self.target_joints11 = [-0.31827438187537016, -0.6892836143920524, 1.59343319606057, -1.1130607207475656, -0.9955524660733452, -2.231133244350551]
        self.target_joints12 = [0.6255435168620961, -0.565093488542824, 1.8872192790804434, -1.9124751009621006, -0.7800579927796214, -0.7312039004787723]
        self.target_joints13 = [0.8098417211466316, -0.6867088702501638, 1.6967894127021093, -1.9337026253789045, -0.9715840162626321, -0.5607360243672357]
        self.target_joints14 = [0.9005572436741096, -0.7925546098787093, 1.507289522176431, -1.9194472566959284, -1.1209649828629207, -0.5078461777504799]
        self.target_joints15 = [1.0130539618090189, -1.009404632167283, 1.176843961487181, -1.966921831552117, -1.4305872869888407, -0.43533416022202065]
        self.target_joints16 = [1.0240855739924541, -1.2559606185190377, 1.1258279150134052, -1.9987021317883595, -1.3172362267762336, -0.3433368582618881]
        self.target_joints17 = [1.0862916907345248, -1.5689582130105242, 0.5292465413220299, -2.0250969876682436, -1.7452062139519677, -0.41759111577547614]
        self.target_joints18 = [1.0862916907345248, -1.5689582130105242, 0.5292465413220299, -2.0250969876682436, -1.7452062139519677, -0.41759111577547614]
        self.target_joints19 = [0.38362542554871515, -0.22737696544543873, 1.9523708519881415, -1.7814992406553491, -0.9384830544418197, -1.8169970474572343]
        self.target_joints20 = [0.46958667289818384, -0.836247495986985, 1.004251948090198, -1.6950759348567521, -1.6543002754065492, -1.813730946696361]
        self.target_joints21 = [0.42947840161715445, -1.3033782017278979, 0.3407980667947149, -1.7480840258259072, -1.9806482964426033, -1.892567971813759]
        self.target_joints22 = [0.20035173933553313, -1.3284284275734013, 0.06314807681963187, -1.6161132085359027, -2.1437259005291693, -1.9856784751095446]
        self.target_joints23 = [1.4880029095620773, -1.0225201679037053, 1.9270894637600195, -2.3622236205841753, -0.9004552446043599, 1.3452106034986446]
        self.target_joints24= [1.1343210020457768, -0.6211748674099304, 2.193103570300328, -2.483423282845354, -0.7001967159069791, 1.0948143388549965]
        self.target_joints25= [0.7861667516880451, -0.3448719043142088, 2.0275439000778084, -1.9693815271349138, -0.7830987901122741, 0.19781747524221377]
        self.target_joints26= [0.02588299631837111, -0.22842145722496898, 1.7281763640453471, -1.3586446533014405, -1.025908774074722, -0.9433726182032043]
        self.target_joints27= [-0.23454779299658668, -0.4596094061903111, 1.473853002058268, -1.112964846948322, -1.1908766899030265, -1.340979373158726]
        self.target_joints28= [-0.2899548594756851, -0.8183531839905251, 1.0073918150154013, -1.1447853609170258, -1.503320879519995, -1.3302116802393185]
        self.target_joints29= [-0.34642825565525115, -1.3676333546126758, 0.10730195611260059, -1.1232734108958011, -2.024664490307215, -1.2334430588701988]
        self.target_joints30= [-0.5262645998494673, -1.4994390559150954, 0.05119847300844633, -0.9041506469329539, -2.035553623372331, -1.3132084619435984]
        # self.target_joints31= []
        # self.target_joints32= []
        # self.target_joints33= []
        # self.target_joints34= []
        # self.target_joints35= []
        # self.target_joints36= []
        
    
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
        elif self.change_pos_count == 16:
            self.go_to_position(self.target_joints16)
        elif self.change_pos_count == 17:
            self.go_to_position(self.target_joints17)
        elif self.change_pos_count == 18:
            self.go_to_position(self.target_joints18)
        elif self.change_pos_count == 19:
            self.go_to_position(self.target_joints19)
        elif self.change_pos_count == 20:
            self.go_to_position(self.target_joints20)
        elif self.change_pos_count == 21:
            self.go_to_position(self.target_joints21)
        elif self.change_pos_count == 22:
            self.go_to_position(self.target_joints22)
        elif self.change_pos_count == 23:
            self.go_to_position(self.target_joints23)
        elif self.change_pos_count == 24:
            self.go_to_position(self.target_joints24)
        elif self.change_pos_count == 25:
            self.go_to_position(self.target_joints25)
        elif self.change_pos_count == 26:
            self.go_to_position(self.target_joints26)
        elif self.change_pos_count == 27:
            self.go_to_position(self.target_joints27)
        elif self.change_pos_count == 28:
            self.go_to_position(self.target_joints28)
        elif self.change_pos_count == 29:
            self.go_to_position(self.target_joints29)
        elif self.change_pos_count == 30:
            self.go_to_position(self.target_joints30)
        # elif self.change_pos_count == 31:
        #     self.go_to_position(self.target_joints31)
        # elif self.change_pos_count == 32:
        #     self.go_to_position(self.target_joints32)
        # elif self.change_pos_count == 33:
        #     self.go_to_position(self.target_joints33)


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
                    print(f"-------------------Position Number {self.change_pos_count}-------------------")
                    self.change_pos()
                    self.save_data()
                elif key == ord('q'):  # 'q' key
                    rospy.loginfo("Quitting program.")
                    break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    calib = KinovaCalibration()
    calib.run()