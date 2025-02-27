#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
import cv2
import pyrealsense2 as rs

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

        if CALIBRATION:
            self.markerLength = 0.066   # 0.1325 #0.066  # 6.6 cm
            self.count = 0
            self.count_limit = 100
        else:
            self.T_ee_camera_color = np.array([[-0.99999149, -0.00346816,  0.00223273,  0.0319287, ],
                                               [ 0.0035009,  -0.9998839,   0.01482983, -0.0723345, ],
                                               [ 0.00218104,  0.01483752,  0.99988754, -0.13790191,],
                                               [ 0. ,         0.,          0.,          1.        ]])

    
    def start(self):
        if CALIBRATION:
            rospy.loginfo("Moving to start position")
            self.go_sp()
            self.gripper_move(0.6)
            rospy.loginfo("Calculate EE to Camera Color Frame")
            self.calculate_ee_camera_color()
            rospy.loginfo("Moving to start position")
            self.go_sp()

        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
        else:
            rospy.loginfo("Calculate EE to Camera Frame and Graspnet Frame")
            T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet = self.calculate_ee_to_camera_graspnet()
            rospy.loginfo("Publishing Camera Frame and Graspnet frame to tf")
            self.publish_tf(T_ee_camera_color, T_ee_camera_depth, T_ee_graspnet)


    def calculate_ee_camera_color(self):
        # --- Get EE to Camera Color ---
        # T_aruco_camera_color = np.array([[-0.04793425,  0.99878315,  0.01159826,  0.02139552,],
        #                                  [ 0.99879475,  0.04805102, -0.01000745,  0.02883001,],
        #                                  [-0.01055258,  0.01110458, -0.99988266,  0.6365618, ],
        #                                  [ 0. ,         0. ,         0. ,         1.        ]])
        T_aruco_camera_color = self.get_aruco_to_camera() # T_aruco_camera
        print("Transformation Matrix (aruco to camera_color):\n", T_aruco_camera_color)

        T_camera_color_aruco =  np.linalg.inv(T_aruco_camera_color) # T_camera_aruco
        print("Transformation Matrix (camera_color to aruco):\n", T_camera_color_aruco)

        T_base_ee = self.get_base_to_ee() # T_base_ee(1)
        print("Transformation Matrix (base to ee = base to ee(1)):\n", T_base_ee)

        print("Moving ee to aruco marker")
        self.go_aruco()

        T_base_aruco = self.get_base_to_aruco() # T_base_aruco = T_base_ee(0)
        print("Transformation Matrix (base to aruco = base to ee(0)):\n", T_base_aruco)

        delta = self.calculate_delta(T_base_ee, T_base_aruco)
        print("delta: ", delta)
        delta_inv = np.linalg.inv(delta)

        print("Moving to start position")
        self.go_sp()

        T_camera_color_ee = T_camera_color_aruco @ delta_inv
        print("Transformation Matrix (camera_color to ee):\n", T_camera_color_ee)
        
        T_ee_camera_color = np.linalg.inv(T_camera_color_ee)
        print("Transformation Matrix (ee to camera_color):\n", T_ee_camera_color)

        self.T_ee_camera_color = T_ee_camera_color

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

                        T_aruco_camera = self.construct_homogeneous_transform(tvec, rotation_matrix)
                        # print("Transformation Matrix (arco to camera):\n", T_aruco_camera)

                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
                # Show frame
                cv2.imshow("RealSense ArUco Pose Estimation", frame)
                cv2.imshow("Enhanced ArUco Detection", enhanced)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.count += 1
                if self.count == self.count_limit:
                    break
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
        
        return T_aruco_camera

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

    def go_aruco(self):
        # self.arm_group.set_joint_value_target([-0.03531032026114289, -1.078645222612689, 
        #                                        1.4896657873467227, -1.6066813572192782, 
        #                                        -0.5534677251201918, -1.5783778137861342])

        self.arm_group.set_joint_value_target([-0.06018797326800929, -1.1773499620942962, 
                                               1.4097070387781836, -1.5753114501070167, 
                                               -0.5708187522542758, -1.6280532249671644])
        self.arm_group.go(wait=True)

    def get_base_to_aruco(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                # (trans_base_aruco, rot_base_aruco) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                (trans_base_aruco, _) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        rot_base_aruco = np.eye(3)
        
        T_base_aruco = self.construct_homogeneous_transform(trans_base_aruco, rot_base_aruco)

        return T_base_aruco
    
    def calculate_delta(self, T_base_ee, T_base_aruco):
        T_base_ee_inv = np.linalg.inv(T_base_ee)
        delta = T_base_ee_inv @ T_base_aruco

        return delta

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