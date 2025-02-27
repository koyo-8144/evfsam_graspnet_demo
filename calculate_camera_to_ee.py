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


class CalculateCameraToEe():
    def __init__(self):
        self.filepath = "/home/chart-admin/koyo_ws/langsam_grasp_ws/src/demo_pkg_v2/src/data/gg_values.txt"
        # self.filepath = "/home/ubuntu/catkin_workspace/src/demo_pkg/data/gg_values.txt"
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


        self.markerLength = 0.066  # 6.6 cm
        self.count = 0
        self.count_limit = 30

    
    def start(self):
        rospy.loginfo("Moving to start position")
        self.go_sp()
        rospy.loginfo("Calculate_camera_to_ee")
        self.calculate_camera_to_ee()

        # self.process_graspnet_output()

    ####### Get Camera to EE Transformation Matrix #######

    def calculate_camera_to_ee(self):
        T_camera_aruco = self.get_camera_to_aruco() # T_camera_aruco
        print("Transformation Matrix (camera to arco):\n", T_camera_aruco)

        T_base_ee = self.get_base_to_ee() # T_base_ee(1)
        print("Transformation Matrix (base to ee = base to ee(1)):\n", T_base_ee)

        print("Moving ee to aruco marker")
        self.go_aruco()

        T_base_aruco = self.get_base_to_aruco() # T_base_aruco = T_base_ee(0)
        print("Transformation Matrix (base to aruco = base to ee(0)):\n", T_base_aruco)

        delta_inv = self.calculate_delta_inv(T_base_aruco, T_base_ee)
        print("Delta Inverse:\n", delta_inv)

        T_camera_ee = T_camera_aruco @ delta_inv
        print("Transformation Matrix (camera to ee):\n", T_camera_ee)

        # Transformation Matrix (camera to ee):
        # [[-0.04498669  0.99839626  0.03436716  0.12331421]
        # [ 0.9986172   0.0458801  -0.02566522  0.040693  ]
        # [-0.02720083  0.03316504 -0.99907967  1.08776383]
        # [ 0.          0.          0.          1.        ]]



    def get_camera_to_aruco(self):
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
        
                # Detect ArUco markers
                detector = cv2.aruco.ArucoDetector(dictionary)
                markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        
                if markerIds is not None:
                    for i in range(len(markerIds)):
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], markerLength, camMatrix, distCoeffs)
                        cv2.drawFrameAxes(frame, camMatrix, distCoeffs, rvec, tvec, 0.05)

                        tvec = tvec[0]
                        print("Translation Vector (tvecs):\n", tvec[i])
                        # print("Rotation Vector (rvecs):\n", rvec[i])
                        rotation_matrix, _ = cv2.Rodrigues(rvec[i])
                        print("Rotation Matrix:\n", rotation_matrix)

                        T_camera_aruco = self.construct_homogeneous_transform(tvec, rotation_matrix)
                        print("Transformation Matrix (camera to arco):\n", T_camera_aruco)

                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
                # Show frame
                cv2.imshow("RealSense ArUco Pose Estimation", frame)
        
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.count += 1
                if self.count == self.count_limit:
                    break
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
        
        return T_camera_aruco

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
        self.arm_group.set_joint_value_target([-0.03531032026114289, -1.078645222612689, 
                                               1.4896657873467227, -1.6066813572192782, 
                                               -0.5534677251201918, -1.5783778137861342])
        self.arm_group.go(wait=True)

    def get_base_to_aruco(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_base_aruco, rot_base_aruco) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_aruco = self.construct_rot_matrix_homogeneous_transform(trans_base_aruco, rot_base_aruco)

        return T_base_aruco

    def calculate_delta_inv(self, T_base_aruco, T_base_ee):
        T_base_aruco_inv = np.linalg.inv(T_base_aruco)
        delta_inv = T_base_aruco_inv @ T_base_ee

        return delta_inv


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

        return trans_base_camera, rot_base_camera
    
    def get_camera_ee(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_camera_ee, rot_camera_ee) = self.listener.lookupTransform('/d435_depth_optical_frame', '/tool_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        return trans_camera_ee, rot_camera_ee

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
        print("trans: ", trans_grasp)
        print("rot: ", rot_grasp.shape)
        breakpoint()
        T_grasp_obj = self.construct_homogeneous_transform(trans_grasp, rot_grasp)
        print("Transformation Matrix (grasp to obj):")
        print(T_grasp_obj)

        trans_zero = np.zeros(3)
        rot_z_90 = self.rotation_matrix_z(90)
        T_camera_grasp = self.construct_homogeneous_transform(trans_zero, rot_z_90)
        print("Transformation Matrix (camera to grasp):")
        print(T_camera_grasp)
        
        T_camera_obj = T_camera_grasp @ T_grasp_obj
        print("Transformation Matrix (camera to obj):")
        print(T_camera_obj)
        
        trans_base_camera, rot_base_camera = self.get_base_camera()
        T_base_camera = self.construct_rot_matrix_homogeneous_transform(trans_base_camera, rot_base_camera)
        print("Transformation Matrix (base to camera):")
        print(T_base_camera)
        
        T_base_obj = T_base_camera @ T_camera_obj
        print("Transformation Matrix (base to obj):")
        print(T_base_obj)


        self.target_pos = self.transformation_to_pose(T_base_obj)
        print("Target Pose: ", self.target_pos)

        self.gripper_width = self.read_gripper_width(self.filepath)
        print("Gripper Width: ", self.gripper_width)

        # breakpoint()


    ####### Grasp Planning #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([0.02754534147079857, -0.3292162455300689, 
                                               0.6239125970105316, -1.5710093796821027, 
                                               -2.1819422621718063, -1.6193681240201974])
        self.arm_group.go(wait=True)





def main():
    rospy.init_node('grasp_planning', anonymous=True)
    calculate_camera_to_ee = CalculateCameraToEe()
    calculate_camera_to_ee.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()