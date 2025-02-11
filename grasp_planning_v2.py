#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState


class GraspPlannerNode():
    def __init__(self):
        self.filepath = "/home/chart-admin/koyo_ws/langsam_grasp_ws/src/demo_pkg_v2/src/data/gg_values.txt"
        self.listener = tf.TransformListener()
        self._rate = rospy.Rate(10)  # Broadcast at 10 Hz

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

        # # Initialize planning group
        # self.robot = moveit_commander.robot.RobotCommander()
        # self.arm_group = moveit_commander.move_group.MoveGroupCommander("arm")
        # self.gripper_group = moveit_commander.move_group.MoveGroupCommander("gripper")


        # Set robot arm's speed and acceleration
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

    
    def start(self):
        rospy.loginfo("Processing GraspNet output")
        self.process_graspnet_output()
        rospy.loginfo("Starting grasp planning...")
        rospy.loginfo("Moving to start position")
        # self.go_sp()
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
        print("Rotation Matrix:")
        print(rotation_matrix)

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
        # # Round small floating-point errors to zero
        # rotation_matrix = np.round(rotation_matrix, decimals=10)
        # print("rotation_matrix: ", rotation_matrix)

        return rotation_matrix

    def get_base_camera(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_base_camera, rot_base_camera) = self.listener.lookupTransform('/base_link', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self._rate.sleep()
        return trans_base_camera, rot_base_camera

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
        """Start broadcasting using values from gg_values.txt."""
        poses = self.read_gg_values(self.filepath)
        # print("Obtained poses from gg_values.txt: ", poses)

        trans_grasp = poses["translation"]
        rot_grasp = poses["rotation"]
        T_grasp_obj = self.construct_homogeneous_transform(trans_grasp, rot_grasp)
        print("Transformation Matrix (grasp to obj): ", T_grasp_obj)

        trans_zero = np.zeros(3)
        rot_z_90 = self.rotation_matrix_z(90)
        T_camera_grasp = self.construct_homogeneous_transform(trans_zero, rot_z_90)
        print("Transformation Matrix (camera to grasp): ", T_camera_grasp)
        
        T_camera_obj = T_camera_grasp @ T_grasp_obj
        print("Transformation Matrix (camera to obj): ", T_camera_obj)
        
        trans_base_camera, rot_base_camera = self.get_base_camera()
        T_base_camera = self.construct_rot_matrix_homogeneous_transform(trans_base_camera, rot_base_camera)
        print("Transformation Matrix (base to camera): ", T_base_camera)
        
        T_base_obj = T_base_camera @ T_camera_obj
        print("Transformation Matrix (base to obj): ", T_base_obj)

        self.target_pos = self.transformation_to_pose(T_base_obj)
        print("Target Pose: ", self.target_pos)

        self.gripper_width = self.read_gripper_width(self.filepath)
        print("Gripper Width: ", self.gripper_width)

    ####### Grasp Planning #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

    def grasp_obj(self):
        """
        Grasp object based on object detection.
        """

        self.arm_group.set_pose_target(self.target_pos)
        self.arm_group.go()
        # self.plan_cartesian_path(self.target_pos)
        # self.target_pose.pose.position.z += 0.05
        # self.plan_cartesian_path(self.target_pose.pose)
        # self.arm_group.set_pose_target(self.target_pose.pose)
        # self.arm_group.go()

    def plan_cartesian_path(self, target_pose):
        """
        Cartesian path planning
        """
        waypoints = []
        waypoints.append(target_pose)  # Add the pose to the list of waypoints

        # Set robot arm's current state as start state
        self.arm_group.set_start_state_to_current_state()

        try:
            # Compute trajectory
            (plan, fraction) = self.arm_group.compute_cartesian_path(
                waypoints,   # List of waypoints
                0.01,        # eef_step (endpoint step)
                0.0,         # jump_threshold
                False        # avoid_collisions
            )

            if fraction < 1.0:
                rospy.logwarn("Cartesian path planning incomplete. Fraction: %f", fraction)

            # Execute the plan
            if plan:
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

    # def gripper_move(self, width):
    #     gripper_joints_state = self.gripper_group.get_current_joint_values()
    #     print("gripper_joints_state: ", gripper_joints_state)
    #     gripper_joints_state[2] = width
    #     self.gripper_group.set_joint_value_target(gripper_joints_state)
    #     self.gripper_group.go()

    def gripper_move(self, width):
        joint_state_msg = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState, timeout=1.0)
        print("joint_state_msg: ", joint_state_msg)

        # Find indices of the gripper joints
        right_finger_index = joint_state_msg.name.index('right_finger_bottom_joint')
        left_finger_index = joint_state_msg.name.index('left_finger_bottom_joint')

        # Set desired width for both gripper fingers
        gripper_joints_state = list(joint_state_msg.position)
        gripper_joints_state[right_finger_index] = width
        gripper_joints_state[left_finger_index] = width

 
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

        # 3. Move gripper based on the grasp width
        self.gripper_move(3.6 * self.gripper_width)

        # 4. Place the object at the target location
        self.place_obj()
        
        self.go_sp()

        rospy.loginfo("Grasp completed successfully.")

def main():
    rospy.init_node('grasp_planning', anonymous=True)
    grasp_planner_node = GraspPlannerNode()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()