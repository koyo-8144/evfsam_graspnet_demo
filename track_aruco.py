#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped
import cv2
import pyrealsense2 as rs


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

        self.markerLength = 0.1325 #0.066  # 6.6 cm
        self.count = 0
        self.count_limit = 10000

    
    def start(self):
        rospy.loginfo("Moving to start position")
        self.go_sp()
        self.gripper_move(0.6)
        rospy.loginfo("Getting target aruco position")
        self.get_target_aruco()
        rospy.loginfo("Tracking aruco marker")
        self.track_aruco()
        rospy.loginfo("Moving to start position")
        self.go_sp()
        
        

    def get_target_aruco(self):
        # --- Get EE to Camera Color ---
        T_aruco_camera_color = self.get_aruco_to_camera() # T_aruco_camera
        print("Transformation Matrix (aruco to camera_color):\n", T_aruco_camera_color)

        T_camera_color_aruco =  np.linalg.inv(T_aruco_camera_color) # T_camera_aruco
        print("Transformation Matrix (camera_color to aruco):\n", T_camera_color_aruco)

        T_base_camera_color = self.get_base_to_camera_color()
        print("Transformation Matrix (base to camera_color):\n", T_base_camera_color)

        T_base_aruco = T_base_camera_color @ T_camera_color_aruco
        print("Transformation Matrix (base to aruco):\n", T_base_aruco)

        self.target_pos = self.transformation_to_pose(T_base_aruco)
        print("Target Pose:\n", self.target_pos)
    

    def get_aruco_to_camera(self):
        # Load ArUco dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline_profile = pipeline.start(config)
        
        # Get camera intrinsics
        profile = pipeline_profile.get_stream(rs.stream.color)
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        camMatrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0, 0, 1]])
        distCoeffs = np.zeros(5)  # Assuming no lens distortion

        print("intrinsics: ", intrinsics)
        print("camMatrix: ", camMatrix)
        print("distCoeffs: ", distCoeffs)
        
        # Marker length (meters)
        markerLength = self.markerLength
        
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert RealSense frame to NumPy array
                frame = np.asanyarray(color_frame.get_data())

                # Flip the images horizontally (left-to-right) and vertically (up-to-down)
                frame = cv2.flip(frame, -1)  # Flip both horizontally and vertically
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # **Enhance white areas while keeping black intact**
                enhanced = cv2.addWeighted(gray, 1.5, np.zeros_like(gray), 0, 50)
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

                # Detect ArUco markers
                detector = cv2.aruco.ArucoDetector(dictionary)
                markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(enhanced)
            
                if markerIds is not None:
                    for i in range(len(markerIds)):
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], markerLength, camMatrix, distCoeffs)
                        cv2.drawFrameAxes(frame, camMatrix, distCoeffs, rvec, tvec, 0.05)

                        tvec = tvec[0]
                        # print("Translation Vector (tvecs):\n", tvec[i])
                        # print("Rotation Vector (rvecs):\n", rvec[i])
                        rotation_matrix, _ = cv2.Rodrigues(rvec[i])
                        # print("Rotation Matrix:\n", rotation_matrix)

                        # T_aruco_camera = self.construct_homogeneous_transform(tvec, rotation_matrix)
                        # print("Transformation Matrix (arco to camera):\n", T_aruco_camera)

                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
                # Show frame
                cv2.imshow("RealSense ArUco Pose Estimation", frame)
                cv2.imshow("Enhanced ArUco Detection", enhanced)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                T_aruco_camera = self.construct_homogeneous_transform(tvec, rotation_matrix)
                # print("Transformation Matrix (arco to camera):\n", T_aruco_camera)

                self.count += 1
                if self.count == self.count_limit:
                    break
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
        
        return T_aruco_camera


    def get_ik_solution(self, target_pose):
        """
        Calls the /my_gen3_lite/compute_ik service to get joint positions for a given end-effector pose.

        Args:
            target_pose (Pose): The desired end-effector pose (position and orientation).

        Returns:
            list: Joint positions if a solution is found, None otherwise.
        """
        # Wait for the service to be available
        rospy.wait_for_service('/my_gen3_lite/compute_ik')

        try:
            # Create a service proxy
            compute_ik = rospy.ServiceProxy('/my_gen3_lite/compute_ik', GetPositionIK)

            # Prepare the IK request
            ik_request = GetPositionIKRequest()
            ik_request.ik_request.group_name = "arm"
            ik_request.ik_request.timeout = rospy.Duration(5.0)
            ik_request.ik_request.avoid_collisions = True

            # Prepare robot state with current joint states
            current_state = self.robot.get_current_state()
            ik_request.ik_request.robot_state = current_state

            # Define the target pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.pose = target_pose
            ik_request.ik_request.pose_stamped = pose_stamped
            ik_request.ik_request.ik_link_name = "tool_frame" # self.arm_group.get_end_effector_link()

            # Call the IK service
            response = compute_ik(ik_request)

            # Check if a solution was found
            if response.error_code.val == 1:
                joint_positions = response.solution.joint_state.position
                joint_names = response.solution.joint_state.name
                print("IK solution found:")
                for name, pos in zip(joint_names, joint_positions):
                    print(f"{name}: {pos}")
                return joint_positions
            else:
                rospy.logerr("Failed to find IK solution.")
                return None

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
  

    def track_aruco(self):
        print("Planning motion .....")
        self.arm_group.set_pose_target(self.target_pos)
        self.arm_group.go()

        # joint_positions = self.get_ik_solution(self.target_pos)
        # joint_positions = np.array(joint_positions[:6])
        # print(joint_positions)
        # self.arm_group.set_joint_value_target(joint_positions)
        # self.arm_group.go(wait=True)

        
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
   

    
    def get_base_to_camera_color(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                # (trans_base_aruco, rot_base_aruco) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                (trans_base_camera_color, rot_base_camera_color) = self.listener.lookupTransform('/base_link', '/camera_color', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_camera_color = self.construct_rot_matrix_homogeneous_transform(trans_base_camera_color, rot_base_camera_color)

        return T_base_camera_color
    

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