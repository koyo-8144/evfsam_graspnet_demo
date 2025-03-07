#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState

import sys
import os
import time
import threading

sys.path.append("")
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

sys.path.append("/home/sandisk/koyo_ws/demo_ws/src/Kinova-kortex2_Gen3_G3L/api_python/examples")
import utilities

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

SET_START = 0
DEBUG_OBJ = 0

class GraspAPIPlannerNode():
    def __init__(self):
        self.filepath = "/home/sandisk/koyo_ws/demo_ws/src/demo_pkg/evfsam_graspnet_demo/data/gg_values.txt"
        # self.filepath = "/home/ubuntu/catkin_workspace/src/demo_pkg/data/gg_values.txt"
        self.listener = tf.TransformListener()
        if DEBUG_OBJ:
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
        self.arm_group.set_max_acceleration_scaling_factor(1.0)  # 50% of max acceleration
        self.arm_group.set_max_velocity_scaling_factor(1.0)      # 50% of max velocity
        self.arm_group.set_planning_time(15.0)  # Increase to 15 seconds for complex paths
        self.arm_group.allow_replanning(True)



        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

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


    
    def start(self):
        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
        else:
            rospy.loginfo("Processing GraspNet output")
            self.process_graspnet_output()
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Starting grasp planning...")
            self.execute_grasp_sequence()



    ####### Process GraspNet output #######
    def read_gg_values(self, filepath):
        """Read translation and rotation matrix from gg_values.txt."""
        print("Start reading gg values")
        with open(filepath, 'r') as file:
            lines = file.readlines()

        poses = {}
        for i, line in enumerate(lines):
            if 'translation:' in line:
                translation_str = line.split('[')[1].split(']')[0]
                translation = [float(num) for num in translation_str.split()]
                poses['translation'] = translation
            elif 'rotation:' in line:
                rotation = [list(map(float, lines[i + j + 1].strip().strip('[]').split())) for j in range(3)]
                poses['rotation'] = np.array(rotation)
        return poses

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

    def get_base_to_camera(self):
        while not rospy.is_shutdown():
            try:
                (trans_base_camera, rot_base_camera) = self.listener.lookupTransform('/base_link', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_camera = self.construct_rot_matrix_homogeneous_transform(trans_base_camera, rot_base_camera)

        return T_base_camera
    
    def get_color_to_depth(self):
        while not rospy.is_shutdown():
            try:
                (trans_color_depth, rot_color_depth) = self.listener.lookupTransform('/d435_color_optical_frame', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_color_depth = self.construct_rot_matrix_homogeneous_transform(trans_color_depth, rot_color_depth)

        return T_color_depth
    
    def get_ee_to_camera_color(self):
        while not rospy.is_shutdown():
            try:
                (trans_ee_camera_color, rot_ee_camera_color) = self.listener.lookupTransform('/tool_frame', '/camera_color', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_ee_camera_color = self.construct_rot_matrix_homogeneous_transform(trans_ee_camera_color, rot_ee_camera_color)

        return T_ee_camera_color
    
    def get_ee_to_camera_depth(self):
        while not rospy.is_shutdown():
            try:
                (trans_ee_camera_depth, rot_ee_camera_depth) = self.listener.lookupTransform('/tool_frame', '/camera_depth', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_ee_camera_depth = self.construct_rot_matrix_homogeneous_transform(trans_ee_camera_depth, rot_ee_camera_depth)

        return T_ee_camera_depth
    
    def get_base_to_camera_color(self):
        while not rospy.is_shutdown():
            try:
                (trans_base_camera_color, rot_base_camera_color) = self.listener.lookupTransform('/base_link', '/camera_color', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_camera_color = self.construct_rot_matrix_homogeneous_transform(trans_base_camera_color, rot_base_camera_color)

        return T_base_camera_color
    
    def get_base_to_graspnet(self):
        while not rospy.is_shutdown():
            try:
                (trans_base_graspnet, rot_base_graspnet) = self.listener.lookupTransform('/base_link', '/graspnet', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_graspnet = self.construct_rot_matrix_homogeneous_transform(trans_base_graspnet, rot_base_graspnet)

        return T_base_graspnet
    
    def get_camera_color_to_camera_depth(self):
        while not rospy.is_shutdown():
            try:
                (trans_camera_color_camera_depth, rot_camera_color_camera_depth) = self.listener.lookupTransform('/camera_color', '/camera_depth', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_camera_color_camera_depth = self.construct_rot_matrix_homogeneous_transform(trans_camera_color_camera_depth, rot_camera_color_camera_depth)

        return T_camera_color_camera_depth
    
    def get_camera_depth_to_graspnet(self):
        while not rospy.is_shutdown():
            try:
                (trans_camera_depth_graspnet, rot_camera_depth_graspnet) = self.listener.lookupTransform('/camera_depth', '/graspnet', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_camera_depth_graspnet = self.construct_rot_matrix_homogeneous_transform(trans_camera_depth_graspnet, rot_camera_depth_graspnet)

        return T_camera_depth_graspnet
    
    def replace_rotation_with_identity(self, T):
        """
        Replaces the rotation part of a 4x4 transformation matrix with an identity matrix.

        Args:
        - T (numpy.ndarray): A 4x4 transformation matrix.

        Returns:
        - numpy.ndarray: A modified 4x4 transformation matrix with an identity rotation.
        """
        # Check if the input is a 4x4 matrix
        if T.shape != (4, 4):
            raise ValueError("Input matrix must be 4x4.")

        # Create a copy to avoid modifying the original matrix
        T_new = T.copy()

        # Replace the rotation part (top-left 3x3) with an identity matrix
        T_new[:3, :3] = np.eye(3)

        # Return the modified matrix
        return T_new

    def replace_orientation_with_tilt(self, transform_matrix, tilt_angle=0):
        """
        Replace the orientation of a 4x4 transformation matrix with a simple tilt around the Z-axis.

        Args:
            transform_matrix (np.ndarray): The original 4x4 transformation matrix.
            tilt_angle (float): The tilt angle in degrees (default: 45°).

        Returns:
            np.ndarray: Modified 4x4 transformation matrix with the new orientation.
        """
        # Convert tilt angle from degrees to radians
        tilt_radians = np.radians(tilt_angle)

        # Create a new rotation matrix for a tilt around the Z-axis
        rotation_matrix_tilt = np.array([
            [np.cos(tilt_radians), -np.sin(tilt_radians), 0],
            [np.sin(tilt_radians),  np.cos(tilt_radians), 0],
            [0,                    0,                     1]
        ])

        # Extract the translation part from the original matrix
        translation = transform_matrix[:3, 3]

        # Create a new 4x4 transformation matrix
        new_transform = np.identity(4)
        new_transform[:3, :3] = rotation_matrix_tilt  # Replace rotation part
        new_transform[:3, 3] = translation            # Keep translation part

        return new_transform


    def transformation_to_pose(self, T_base_obj):
        # Extract translation (position)
        translation = T_base_obj[:3, 3]  # [x, y, z]

        # Extract rotation matrix
        rotation_matrix = T_base_obj[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = tf_trans.quaternion_from_matrix(T_base_obj)

        # Create a Pose message
        pose = Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def read_gripper_width(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        width = None
        for line in lines:
            if 'width:' in line:
                width = float(line.split('width:')[1].split(',')[0].strip())
                break  # Stop searching once width is found
        return width

    def process_graspnet_output(self):
        # --- Process Graspnet output---
        poses = self.read_gg_values(self.filepath)
        trans_graspnet = poses["translation"]
        rot_graspnet = poses["rotation"]
        print("trans: ", trans_graspnet)
        print("rot: ", rot_graspnet)
        T_graspnet_obj = self.construct_homogeneous_transform(trans_graspnet, rot_graspnet)
        print("Transformation Matrix (graspnet to obj):\n", T_graspnet_obj)

        # --- Get Base to Obj ---
        T_base_graspnet = self.get_base_to_graspnet()
        T_base_obj = T_base_graspnet @ T_graspnet_obj
        print("Transformation Matrix (base to obj):\n", T_base_obj)

        T_base_obj_new = self.replace_orientation_with_tilt(T_base_obj)
        print("Transformation Matrix (base to objj new):\n", T_base_obj_new)

        if DEBUG_OBJ:
            print("Debug Object")
            self.publish_tf(T_base_obj)

        self.target_pos = self.transformation_to_pose(T_base_obj)
        print("Target Pose:\n", self.target_pos)

        self.gripper_width = self.read_gripper_width(self.filepath)
        print("Gripper Width:\n", self.gripper_width)

        # breakpoint()


    ####### Grasp Planning #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([0.02754534147079857, -0.3292162455300689, 
                                               0.6239125970105316, -1.5710093796821027, 
                                               -2.1819422621718063, -1.6193681240201974])
        self.arm_group.go(wait=True)
    
    # Create closure to set an event after an END or an ABORT
    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def grasp_obj(self):
        """
        Grasp object based on object detection using Kortex API for Cartesian movement.
        Stops ROS before executing Kortex API commands.
        """
        print("Stopping ROS to execute Kortex API commands...")
        rospy.signal_shutdown("Shutting down ROS for Kortex API compatibility.")
        time.sleep(2)  # Wait for ROS to shut down completely

        print("Starting Cartesian action movement with Kortex API...")

        # Create Kortex API clients for base and cyclic control
        args = utilities.parseConnectionArguments()
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            # Create a session to take control of the arm
            session_manager = utilities.SessionManager(router)
            session_manager.CreateSession()  # Acquire control

            try:
                # Set the servoing mode to SINGLE_LEVEL_SERVOING
                print("Setting servoing mode to SINGLE_LEVEL_SERVOING...")
                servoing_mode = Base_pb2.ServoingModeInformation()
                servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                base.SetServoingMode(servoing_mode)
                print("Servoing mode set successfully.")

                # Create an action for Cartesian motion
                action = Base_pb2.Action()
                action.name = "Grasp Object Cartesian Movement"
                action.application_data = ""

                feedback = base_cyclic.RefreshFeedback()

                # Set the target position and orientation using Kortex API
                cartesian_pose = action.reach_pose.target_pose
                cartesian_pose.x = self.target_pos.position.x         # X position (meters)
                cartesian_pose.y = self.target_pos.position.y         # Y position (meters)
                cartesian_pose.z = self.target_pos.position.z         # Z position (meters)

                # Extract quaternion from target pose
                quat_x = self.target_pos.orientation.x
                quat_y = self.target_pos.orientation.y
                quat_z = self.target_pos.orientation.z
                quat_w = self.target_pos.orientation.w

                # Convert quaternion to Euler angles for Kortex API
                euler = tf_trans.euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
                cartesian_pose.theta_x = np.degrees(euler[0])  # Roll (degrees)
                cartesian_pose.theta_y = np.degrees(euler[1])  # Pitch (degrees)
                cartesian_pose.theta_z = np.degrees(euler[2])  # Yaw (degrees)

                print(f"Target Cartesian Pose:\n X: {cartesian_pose.x}, Y: {cartesian_pose.y}, Z: {cartesian_pose.z}")
                print(f"Orientation (degrees):\n Theta X: {cartesian_pose.theta_x}, Theta Y: {cartesian_pose.theta_y}, Theta Z: {cartesian_pose.theta_z}")

                # Execute the Cartesian motion
                e = threading.Event()
                notification_handle = base.OnNotificationActionTopic(
                    self.check_for_end_or_abort(e),
                    Base_pb2.NotificationOptions()
                )

                print("Executing Cartesian action...")
                base.ExecuteAction(action)

                print("Waiting for movement to finish...")
                finished = e.wait(TIMEOUT_DURATION)
                base.Unsubscribe(notification_handle)

                if finished:
                    print("Cartesian movement completed successfully.")
                else:
                    print("Timeout on Cartesian action notification wait")

            except Exception as e:
                print(f"Error during Kortex API execution: {e}")
            finally:
                # Release the session after execution
                session_manager.CloseSession()
                print("Control released successfully.")

            return finished

    
    def plan_cartesian_path(self, target_pose):
        """
        Cartesian path planning
        """
        waypoints = []

        # Create intermediate waypoints for smoother path
        current_pose = self.arm_group.get_current_pose().pose
        waypoints.append(current_pose)  # Start from the current pose
        waypoints.append(target_pose)   # Move to the target pose

        # Set robot arm's current state as start state
        self.arm_group.set_start_state_to_current_state()

        try:
            # Compute trajectory with updated parameters
            (plan, fraction) = self.arm_group.compute_cartesian_path(
                waypoints,          # List of waypoints
                0.02,               # eef_step: Increase for faster planning (was 0.01)
                0.1,                # jump_threshold: Allow small jumps (was 0.0)
                # avoid_collisions=True  # Enable collision avoidance
            )

            if fraction < 1.0:
                rospy.logwarn("Cartesian path planning incomplete. Fraction: %f", fraction)

            # Execute the plan if it is valid
            if plan and len(plan.joint_trajectory.points) > 0:
                rospy.loginfo("Executing Cartesian path...")
                self.arm_group.execute(plan, wait=True)
            else:
                rospy.logerr("Failed to compute Cartesian path")

        except Exception as e:
            rospy.logerr("Error in Cartesian path planning: %s", str(e))

    def place_obj(self):
        self.target_pos.position.z += 0.07
        # self.plan_cartesian_path(self.target_pose.pose)

        self.arm_group.set_joint_value_target([-1.1923993012061151, 0.7290586635521652,
                                               -0.7288901499177471, 1.6194515338395425,
                                               -1.6699862200379725, 0.295133228129065])
        self.arm_group.go()

        self.gripper_move(0.6)

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

    def execute_grasp_sequence(self):
        """
        Execute the grasp sequence as a separate method.
        This method waits for the object pose to be detected before proceeding.
        """
        rospy.loginfo('Starting grasp sequence')

        self.go_sp()
        self.gripper_move(0.6)

        # 1. Grasp the object
        self.grasp_obj()

        # # 3. Move gripper based on the grasp width
        # self.gripper_move(3.6 * self.gripper_width)

        # # 4. Place the object at the target location
        # self.place_obj()
        
        # self.go_sp()

        rospy.loginfo("Grasp completed successfully.")


    ####### Debug Obj #######
    def publish_tf(self, T_base_obj):
        """ Continuously broadcast the transformation """
        # Extract translation (last column of the first 3 rows)
        translation_base_obj = T_base_obj[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix_base_obj = T_base_obj[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion_base_obj = tf.transformations.quaternion_from_matrix(T_base_obj)

        rospy.loginfo("Publishing transformation: base_link → obj")
        rospy.loginfo(f"Translation: {translation_base_obj}")
        rospy.loginfo(f"Quaternion: {quaternion_base_obj}")

        while not rospy.is_shutdown():
            # Broadcast transform (base_link → obj)
            self.broadcaster.sendTransform(
                translation_base_obj,    # Position (x, y, z)
                quaternion_base_obj,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                "obj",  # Child frame
                "base_link"      # Parent frame
            )

            self.rate.sleep()

def main():
    rospy.init_node('grasp_api_planning', anonymous=True)
    grasp_api_planner_node = GraspAPIPlannerNode()
    grasp_api_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()