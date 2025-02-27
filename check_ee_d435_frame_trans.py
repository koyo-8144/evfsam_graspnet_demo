#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
import cv2

SET_START = 0

class GraspPlannerNode():
    def __init__(self):
        self.filepath = "/home/chart-admin/koyo_ws/langsam_grasp_ws/src/demo_pkg_v2/src/data/gg_values.txt"
        # self.filepath = "/home/ubuntu/catkin_workspace/src/demo_pkg/data/gg_values.txt"
        self.listener = tf.TransformListener()

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
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Starting checking translation between ee and d435depth")
            self.check_ee_d435_trans()
            rospy.loginfo("Moving to start position")
            self.go_sp()
        
    def check_ee_d435_trans(self):
        error_x_list = []
        error_y_list = []
        error_z_list = []

        for i in range(10):
            print("Getting values")
            T_base_ee_up = self.get_base_to_ee()
            T_ee_d435detph_up = self.get_ee_to_d435depth()
            T_base_d435depth_up = self.get_base_to_d435depth()

            print("Going Down")
            self.go_down()

            print("Getting values")
            T_base_ee_down = self.get_base_to_ee()
            T_ee_d435detph_down = self.get_ee_to_d435depth()
            T_base_d435depth_down = self.get_base_to_d435depth()

            print("Calculating delta trans")
            T_base_ee_delta = T_base_ee_up - T_base_ee_down
            T_ee_d435detph_delta = T_ee_d435detph_up - T_ee_d435detph_down
            T_base_d435depth_delta = T_base_d435depth_up - T_base_d435depth_down

            T_base_ee_delta_trans = T_base_ee_delta[:3, 3]
            T_ee_d435detph_delta_trans = T_ee_d435detph_delta[:3, 3]
            T_base_d435depth_delta_trans = T_base_d435depth_delta[:3, 3]

            print("T_base_ee_delta_trans + T_ee_d435detph_delta_trans:\n", T_base_ee_delta_trans + T_ee_d435detph_delta_trans)
            print("T_base_d435depth_delta_trans:\n", T_base_d435depth_delta_trans)
            # T_base_ee_delta_trans + T_ee_d435detph_delta_trans:
            # [-9.81477784e-02  2.43397749e-04  4.96638512e-01]
            # T_base_d435depth_delta_trans:
            # [-9.98750077e-02  6.16993390e-05  4.97805088e-01]

            left = T_base_ee_delta_trans + T_ee_d435detph_delta_trans
            right = T_base_d435depth_delta_trans

            error = abs(left - right)
            print("Error: ", error)
            error_x_list.append(error[0])
            error_y_list.append(error[1])
            error_z_list.append(error[2])


            self.go_sp()
        
        error_x_mean = np.mean(error_x_list)
        error_y_mean = np.mean(error_y_list)
        error_z_mean = np.mean(error_z_list)
        print("Error X Mean: ", error_x_mean)
        print("Error Y Mean: ", error_y_mean)
        print("Error Z Mean: ", error_z_mean)
        # Error X Mean:  0.001878890048088433
        # Error Y Mean:  0.00033492874193615516
        # Error Z Mean:  0.001287196205348623
        


    def get_base_to_ee(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_base_ee, rot_base_ee) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        T_base_ee = self.construct_rot_matrix_homogeneous_transform(trans_base_ee, rot_base_ee)

        return T_base_ee

    def get_ee_to_d435depth(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_ee_d435depth, rot_ee_d435depth) = self.listener.lookupTransform('/tool_frame', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
             
        T_ee_d435depth = self.construct_rot_matrix_homogeneous_transform(trans_ee_d435depth, rot_ee_d435depth)

        return T_ee_d435depth
    
    def get_base_to_d435depth(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_base_d435depth, rot_base_d435depth) = self.listener.lookupTransform('/base_link', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
             
        T_base_d435depth = self.construct_rot_matrix_homogeneous_transform(trans_base_d435depth, rot_base_d435depth)

        return T_base_d435depth

    def go_down(self):
        # self.arm_group.set_joint_value_target([-0.03531032026114289, -1.078645222612689, 
        #                                        1.4896657873467227, -1.6066813572192782, 
        #                                        -0.5534677251201918, -1.5783778137861342])

        self.arm_group.set_joint_value_target([-0.06018797326800929, -1.1773499620942962, 
                                               1.4097070387781836, -1.5753114501070167, 
                                               -0.5708187522542758, -1.6280532249671644])
        self.arm_group.go(wait=True)

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
    

  


    ####### Grasp Planning #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([0.02754534147079857, -0.3292162455300689, 
                                               0.6239125970105316, -1.5710093796821027, 
                                               -2.1819422621718063, -1.6193681240201974])
        self.arm_group.go(wait=True)

    def grasp_obj(self):
        """
        Grasp object based on object detection.
        """

        # self.target_pos.position.z += 0.05
        # self.target_pos.position.x += 0.125
        # self.target_pos.position.y -= 0.2
        # self.target_pos.position.x += 0.1
        # self.target_pos.position.y -= 0.15
        # self.target_pos.position.x += 0.075
        # self.target_pos.position.y += 0.075
        self.arm_group.set_pose_target(self.target_pos)
        self.arm_group.go()
        # self.plan_cartesian_path(self.target_pos)
        # self.target_pos.position.z += 0.05
        # self.plan_cartesian_path(self.target_pos)
        # self.arm_group.set_pose_target(self.target_pos)
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

        # 3. Move gripper based on the grasp width
        self.gripper_move(3.6 * self.gripper_width)

        # 4. Place the object at the target location
        self.place_obj()
        
        self.go_sp()

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
    rospy.init_node('grasp_planning', anonymous=True)
    grasp_planner_node = GraspPlannerNode()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()