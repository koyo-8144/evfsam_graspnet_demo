#!/usr/bin/env python3
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import time
from moveit_msgs.msg import Constraints
import paho.mqtt.client as mqtt
from sensor_msgs.msg import  JointState
 
# Define the broker address and port
BROKER_ADDRESS = "172.22.247.120"
BROKER_PORT = 1883


class GraspPlannerNode():
    def __init__(self):
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
        self.arm_group.clear_path_constraints()

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
        

        # Targe grasp position
        self.target_pose = PoseStamped()
        self.target_pose_update = True

        # Create an MQTT client instance
        self.client = mqtt.Client()
        
        # Attach callback functions
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.object = None

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

    

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to Mosquitto broker!")
            client.subscribe("robot/arrival")
            print("Waiting for message ....")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        print(f"Message received: Topic = {msg.topic}, Payload = {msg.payload.decode()}")
        self.object = msg.payload.decode()
        print("Received object name: ", self.object)

        # Disconnect client after receiving message
        self.client.loop_stop()

    def receive_message(self):
        print(f"Connecting to broker {BROKER_ADDRESS}:{BROKER_PORT}...")
        self.client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

        # Start loop in a background thread
        self.client.loop_start()
        print("Listening for messages...")

        # Wait for a message to be received
        while self.object is None:
            time.sleep(0.1)

        print("Message received. Exiting receive_message.")

    def send_message(self):
        print(f"Connecting to broker {BROKER_ADDRESS}:{BROKER_PORT}...")
        self.client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

        # Start loop in a background thread
        self.client.loop_start()

        # Publish a message
        print("Publishing message...")
        self.client.publish("robot/arrival", "Grasp done")

        # Wait for the message to be sent
        time.sleep(1)

        # Stop the loop and disconnect
        self.client.loop_stop()
        self.client.disconnect()
        print("Message sent and client disconnected.")

    ####### grasp planning #######

    def start_grasp_planning(self):
        rospy.loginfo("Starting grasp planning...")
        rospy.loginfo("Moving to start position")
        self.go_sp()
        self.receive_message()
        input("Waiting for input ....")
        self.execute_grasp_sequence()
        self.send_message()

    def execute_grasp_sequence(self):
        """
        Execute the grasp sequence as a separate method.
        This method waits for the object pose to be detected before proceeding.
        """


        rospy.loginfo('Starting grasp sequence')

        # self.object = "drink"

        if self.object == "banana":
            # 1. Grasp the object
            self.grasp_banana()

            # 3. Move gripper based on the grasp width
            width = 0.05
            self.gripper_move(3.6 * width)

            self.go_safepos()

        elif self.object == "orange":

            # 1. Grasp the object
            self.grasp_orange()

            # 3. Move gripper based on the grasp width
            width = 0.06
            self.gripper_move(3.6 * width)
        
            self.go_safepos()

        elif self.object == "drink":

            # 1. Grasp the object
            self.grasp_drink()

            # 3. Move gripper based on the grasp width
            width = 0.06
            self.gripper_move(3.6 * width)
        
            self.go_safepos()

        # 4. Place the object at the target location
        self.place_obj()
        
        self.go_sp()

        rospy.loginfo("Grasp completed successfully.")


    def grasp_banana(self):

        self.arm_group.set_joint_value_target([0.2710410894139424, -1.670172715503421, -0.00410393123981212, -1.5096645292365638, -1.4751765931200378, -1.2731055169847494])
        self.arm_group.go(wait=True)

        # pose_goal =Pose()
        # pose_goal.position.x = 0.5220786750243936
        # pose_goal.position.y = 0.09561009672562862
        # pose_goal.position.z = -0.04516127388096612
        # pose_goal.orientation.x = -0.6943437946518083
        # pose_goal.orientation.y = 0.7188652322922656
        # pose_goal.orientation.z = 0.021450509382198883
        # pose_goal.orientation.w = 0.025677777885934967
        # self.arm_group.set_pose_target(pose_goal)
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = self.arm_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # self.arm_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # self.arm_group.clear_pose_targets()
    
    def grasp_orange(self):

        self.arm_group.set_joint_value_target([-0.4891768858768071, -1.482970067734045, 0.42527939552642646, -1.5665448564306939, -1.2400008267384104, -2.0606909345854794])
        self.arm_group.go(wait=True)

        # pose_goal =Pose()
        # pose_goal.position.x = 0.5220786750243936
        # pose_goal.position.y = 0.09561009672562862
        # pose_goal.position.z = -0.04516127388096612
        # pose_goal.orientation.x = -0.6943437946518083
        # pose_goal.orientation.y = 0.7188652322922656
        # pose_goal.orientation.z = 0.021450509382198883
        # pose_goal.orientation.w = 0.025677777885934967
        # self.arm_group.set_pose_target(pose_goal)
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = self.arm_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # self.arm_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # self.arm_group.clear_pose_targets()
    
    def grasp_drink(self):

        self.arm_group.set_joint_value_target([0.12061962577575984, -1.0196594002607426, 1.7749393422582664, -2.262562273639012, -0.32320016462757994, 0.22415667105531542])
        self.arm_group.go(wait=True)
    
    def go_safepos(self):
        self.arm_group.set_joint_value_target([0.9311649540823903, -0.8526062292989058, 1.1426484404583457, -1.475054620342112, -1.1986110423408354, -0.6476030128034402])

        self.arm_group.go(wait=True)

        # waypoint_pose =Pose()
        # waypoint_pose.position.x = 0.403037475480482
        # waypoint_pose.position.y = 0.3234225897779779
        # waypoint_pose.position.z = 0.15802923128916288
        # waypoint_pose.orientation.x = -0.7076428207095652
        # waypoint_pose.orientation.y = 0.7062677469214905
        # waypoint_pose.orientation.z = -0.005752896205189294
        # waypoint_pose.orientation.w = 0.019859812232360337
        # self.arm_group.set_pose_target(waypoint_pose)
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = self.arm_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # self.arm_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # self.arm_group.clear_pose_targets()

    def plan_cartesian_path(self, target_pose, waypoint_pose):
        """
        Cartesian path planning
        """
        from geometry_msgs.msg import Pose

        waypoints = []

        # Add waypoint poses
        if isinstance(waypoint_pose, Pose):
            waypoints.append(waypoint_pose)
        elif isinstance(waypoint_pose, PoseStamped):
            waypoints.append(waypoint_pose.pose)
        else:
            rospy.logerr("Invalid waypoint_pose type. Must be Pose or PoseStamped.")
            return

        if isinstance(target_pose, Pose):
            waypoints.append(target_pose)
        elif isinstance(target_pose, PoseStamped):
            waypoints.append(target_pose.pose)
        else:
            rospy.logerr("Invalid target_pose type. Must be Pose or PoseStamped.")
            return

        rospy.loginfo(f"Waypoints: {waypoints}")

        # Set robot arm's current state as start state
        self.arm_group.set_start_state_to_current_state()

        try:
            # # Compute Cartesian path
            # (plan, fraction) = self.arm_group.compute_cartesian_path(
            #     waypoints,   # List of Pose objects
            #     0.01,        # eef_step (endpoint step size in meters)
            #     0.0          # jump_threshold
            # )
            # Compute Cartesian path
            (plan, fraction) = self.arm_group.compute_cartesian_path(
                waypoints,   # List of Pose objects
                0.01,        # eef_step (endpoint step size in meters)
                True,
                None
            )

            if fraction < 1.0:
                rospy.logwarn("Cartesian path planning incomplete. Fraction: %f", fraction)

            # Execute the plan if planning was successful
            if plan:
                self.arm_group.execute(plan, wait=True)
            else:
                rospy.logerr("Failed to compute Cartesian path")

        except Exception as e:
            rospy.logerr(f"Error in Cartesian path planning: {str(e)}")


    def read_gripper_width(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        width = None
        for line in lines:
            if 'width:' in line:
                width = float(line.split('width:')[1].split(',')[0].strip())
                break  # Stop searching once width is found
        return width

    def place_obj(self):
        # self.arm_group.set_joint_value_target([1.3848293857713174, -2.001753315449263, -1.031860406478831, -1.8942212622184798, -1.4528502484374677, -1.0467149864070748])
        self.arm_group.set_joint_value_target([1.2601399172176984, -1.8302055275156999, -0.6814011901976365, -1.781343712047689, -1.2712748600514292, -0.9486067950096464])
        self.arm_group.go()


        self.gripper_move(0.6)

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

        self.gripper_move(0.6)

    # def gripper_move(self, width):
    #     gripper_joints_state = self.gripper_group.get_current_joint_values()
    #     print("Current Gripper State: ", gripper_joints_state)
    #     gripper_joints_state[2] = width
    #     self.gripper_group.set_joint_value_target(gripper_joints_state)
    #     self.gripper_group.go()

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
    rospy.init_node('predefined_grasp_planning', anonymous=True)
    grasp_planner_node = GraspPlannerNode()
    grasp_planner_node.start_grasp_planning()
    rospy.spin()
 
if __name__ == "__main__":
    main()