#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
import yaml

CALIBRATION = 0
SET_START = 0


class CameraTFBroadcaster:
    def __init__(self):
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.rate = rospy.Rate(10)  # Broadcast at 10 Hz

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
        # breakpoint()

        self.calib_result_file = FILEPATH

    
    def start(self):
        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
        else:
            rospy.loginfo("Extracting T_ee_camera_color from calibration file")
            self.T_ee_camera_color = self.load_homogeneous_matrix_from_yaml(self.calib_result_file)
            print("Transformation Matrix (ee to camera_color):\n", self.T_ee_camera_color)
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Calculate EE to Camera Frame and Graspnet Frame")
            T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet = self.calculate_ee_to_camera_graspnet()
            rospy.loginfo("Publishing Camera Frame and Graspnet frame to tf")
            self.publish_tf(T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet)


    def load_homogeneous_matrix_from_yaml(self, file_path):
        """
        Reads a YAML file (eMc.yaml) and extracts the homogeneous transformation matrix.

        Args:
            file_path (str): Path to the eMc.yaml file.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix.
        """
        try:
            # Open and load the YAML file
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            # Extract translation (tx, ty, tz)
            tx, ty, tz = data["data"][:3]
            # Extract rotation in theta-U representation (rx, ry, rz)
            rx, ry, rz = data["data"][3:]

            # Convert theta-U representation to rotation matrix
            theta = np.linalg.norm([rx, ry, rz])
            if theta < 1e-10:
                R = np.eye(3)  # Identity matrix if no rotation
            else:
                u = np.array([rx, ry, rz]) / theta
                ux, uy, uz = u
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                one_minus_cos = 1 - cos_theta

                # Rodrigues' rotation formula
                R = np.array([
                    [cos_theta + ux**2 * one_minus_cos, ux * uy * one_minus_cos - uz * sin_theta, ux * uz * one_minus_cos + uy * sin_theta],
                    [uy * ux * one_minus_cos + uz * sin_theta, cos_theta + uy**2 * one_minus_cos, uy * uz * one_minus_cos - ux * sin_theta],
                    [uz * ux * one_minus_cos - uy * sin_theta, uz * uy * one_minus_cos + ux * sin_theta, cos_theta + uz**2 * one_minus_cos]
                ])

            # Construct the 4x4 homogeneous transformation matrix
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R  # Rotation part
            homogeneous_matrix[:3, 3] = [tx, ty, tz]  # Translation part

            return homogeneous_matrix

        except Exception as e:
            print(f"Error loading transformation matrix from {file_path}: {e}")
            return None


    def calculate_ee_to_camera_graspnet(self):
        # --- Get EE to Camera Depth ---
        T_d435color_d435depth = self.get_d435color_to_d435depth()
        print("Transformation Matrix (d435color to d435depth), R should be identity matrix:\n", T_d435color_d435depth)

        T_ee_camera_color = self.T_ee_camera_color
        T_ee_camera_depth = T_ee_camera_color @ T_d435color_d435depth
        print("Transformation Matrix (ee to camera_depth):\n", T_ee_camera_depth)

        # --- Get EE to Graspnet ---
        trans_zero = np.zeros(3)
        rot_z_90 = self.rotation_matrix_z(90)
        T_camera_depth_graspnet = self.construct_homogeneous_transform(trans_zero, rot_z_90)
        print("Transformation Matrix (camera_depth to graspnet):\n", T_camera_depth_graspnet)

        T_ee_graspnet = T_ee_camera_depth @ T_camera_depth_graspnet
        print("Transformation Matrix (ee to graspnet):\n", T_ee_graspnet)


        # --- Get Base to Camera Color, Base to Camera Depth and Base to Graspnet ---
        T_base_ee = self.get_base_to_ee()
        print("Transformation Matrix (base to ee):\n", T_base_ee)

        T_base_camera_color = T_base_ee @ T_ee_camera_color
        print("Transformation Matrix (base to camera_color):\n", T_base_camera_color)

        T_base_camera_depth = T_base_ee @ T_ee_camera_depth
        print("Transformation Matrix (base to camera_depth):\n", T_base_camera_depth)

        T_base_graspnet = T_base_ee @ T_ee_graspnet
        print("Transformation Matrix (base to graspnet):\n", T_base_graspnet)

        return T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet


    def construct_rot_matrix_homogeneous_transform(self, translation, quaternion):
        # Convert quaternion to rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
        # print("Rotation Matrix:")
        # print(rotation_matrix)

        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T
    
    def construct_homogeneous_transform(self, translation, rotation_matrix):
        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T

    def get_d435color_to_d435depth(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_d435color_d435depth, rot_d435color_d435depth) = self.listener.lookupTransform('/d435_color_optical_frame', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
             
        T_d435color_d435depth = self.construct_rot_matrix_homogeneous_transform(trans_d435color_d435depth, rot_d435color_d435depth)

        return T_d435color_d435depth

    def rotation_matrix_z(self, theta_degrees):
        """
        Generates a 3x3 rotation matrix for a rotation around the z-axis.

        Args:
        - theta_degrees (float): The angle of rotation in degrees.

        Returns:
        - numpy.ndarray: A 3x3 rotation matrix.
        """
        # Convert degrees to radians
        theta_radians = np.radians(theta_degrees)

        # Calculate cosine and sine of the angle
        cos_theta = np.cos(theta_radians)
        sin_theta = np.sin(theta_radians)

        # Construct the rotation matrix
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ])

        return rotation_matrix

    def publish_tf(self, T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet):
        """ Continuously broadcast the transformation """
        # Extract translation (last column of the first 3 rows)
        translation_ee_camera_color = T_ee_camera_color[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix_ee_camera_color = T_ee_camera_color[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion_ee_camera_color = tf.transformations.quaternion_from_matrix(T_ee_camera_color)

        # Extract translation (last column of the first 3 rows)
        translation_ee_camera_depth = T_ee_camera_depth[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix_ee_camera_depth = T_ee_camera_depth[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion_ee_camera_depth = tf.transformations.quaternion_from_matrix(T_ee_camera_depth)

        # Extract translation (last column of the first 3 rows)
        translation_ee_graspnet = T_ee_graspnet[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix_ee_graspnet = T_ee_graspnet[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion_ee_graspnet = tf.transformations.quaternion_from_matrix(T_ee_graspnet)

        rospy.loginfo("Publishing transformation: tool_frame → camera_color")
        rospy.loginfo(f"Translation: {translation_ee_camera_color}")
        rospy.loginfo(f"Quaternion: {quaternion_ee_camera_color}")

        rospy.loginfo("Publishing transformation: tool_frame → camera_depth")
        rospy.loginfo(f"Translation: {translation_ee_camera_depth}")
        rospy.loginfo(f"Quaternion: {quaternion_ee_camera_depth}")

        rospy.loginfo("Publishing transformation: tool_frame → graspnet")
        rospy.loginfo(f"Translation: {translation_ee_graspnet}")
        rospy.loginfo(f"Quaternion: {quaternion_ee_graspnet}")

        while not rospy.is_shutdown():
            # Broadcast transform (tool_frame → camera_color)
            self.broadcaster.sendTransform(
                translation_ee_camera_color,    # Position (x, y, z)
                quaternion_ee_camera_color,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                "camera_color",  # Child frame
                # "base_link"      # Parent frame
                 "tool_frame"      # Parent frame
            )

            # Broadcast transform (tool_frame → camera_depth)
            self.broadcaster.sendTransform(
                translation_ee_camera_depth,    # Position (x, y, z)
                quaternion_ee_camera_depth,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                "camera_depth",  # Child frame
                # "base_link"      # Parent frame
                 "tool_frame"      # Parent frame
            )

            # Broadcast transform (tool_frame → camera_depth)
            self.broadcaster.sendTransform(
                translation_ee_graspnet,    # Position (x, y, z)
                quaternion_ee_graspnet,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                "graspnet",  # Child frame
                # "base_link"      # Parent frame
                 "tool_frame"      # Parent frame
            )

            self.rate.sleep()

    ####### Grasp Planning #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([0.02754534147079857, -0.3292162455300689, 
                                               0.6239125970105316, -1.5710093796821027, 
                                               -2.1819422621718063, -1.6193681240201974])
        self.arm_group.go(wait=True)


    def gripper_move(self, width):
        joint_state_msg = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState, timeout=1.0)
        # print("joint_state_msg: ", joint_state_msg)

        # Find indices of the gripper joints
        right_finger_bottom_index = joint_state_msg.name.index('right_finger_bottom_joint')
        # print("right finger bottom index: ", right_finger_bottom_index)

        # self.gripper_group.set_joint_value_target([width])
        self.gripper_group.set_joint_value_target(
            {"right_finger_bottom_joint": width})
        self.gripper_group.go()

 

def main():
    rospy.init_node('camera_tf_broadcaster', anonymous=True)
    tf_broadcaster = CameraTFBroadcaster()
    try:
        tf_broadcaster.start()
    except rospy.ROSInterruptException:
        pass
 
if __name__ == "__main__":
    main()