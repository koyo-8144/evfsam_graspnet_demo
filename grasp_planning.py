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

CALIBRATION = 1
SET_START = 0


class GraspPlannerNode():
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

        if CALIBRATION:
            self.markerLength = 0.066  # 6.6 cm
            self.count = 0
            self.count_limit = 100
        else:
            self.T_ee_camera = np.array([[ 0.03193068,  0.9919599,   0.12245809,  0.03375562,],
                                          [ 0.99915173, -0.02849164, -0.02973286, -0.0781098, ],
                                          [-0.02600477,  0.1233036,  -0.99202821,  1.13342263,],
                                          [ 0.,          0.,          0.,          1.        ]])

    
    def start(self):
        if CALIBRATION:
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Calculate EE to Camera Transformation")
            self.calculate_ee_to_camera()
            self.go_sp()

        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
        else:
            rospy.loginfo("Processing GraspNet output")
            self.process_graspnet_output()
            rospy.loginfo("Starting grasp planning...")
            rospy.loginfo("Moving to start position")
            self.go_sp()
            self.execute_grasp_sequence()


    ####### Get EE to Camera Transformation Matrix #######

    def calculate_ee_to_camera(self):
        T_aruco_camera = self.get_aruco_to_camera() # T_aruco_camera
        print("Transformation Matrix (aruco to camera):\n", T_aruco_camera)

        T_camera_aruco =  np.linalg.inv(T_aruco_camera) # T_camera_aruco
        print("Transformation Matrix (camera to aruco):\n", T_camera_aruco)

        T_base_ee = self.get_base_to_ee() # T_base_ee(1)
        print("Transformation Matrix (base to ee = base to ee(1)):\n", T_base_ee)

        print("Moving ee to aruco marker")
        self.go_aruco()

        T_base_aruco = self.get_base_to_aruco() # T_base_aruco = T_base_ee(0)
        print("Transformation Matrix (base to aruco = base to ee(0)):\n", T_base_aruco)


        delta = self.calculate_delta(T_base_ee, T_base_aruco)
        print("delta: ", delta)
        delta_inv = np.linalg.inv(delta)

        T_camera_ee = T_camera_aruco @ delta_inv
        print("Transformation Matrix (camera to ee):\n", T_camera_ee)
        
        T_ee_camera = np.linalg.inv(T_camera_ee)
        print("Transformation Matrix (ee to camera):\n", T_ee_camera)

        # delta_inv = self.calculate_delta_inv(T_base_aruco, T_base_ee)
        # print("Delta Inverse:\n", delta_inv)

        # T_camera_ee = T_camera_aruco @ delta_inv
        # print("Transformation Matrix (camera to ee):\n", T_camera_ee)

        # T_ee_camera = np.linalg.inv(T_camera_ee)
        # print("Transformation Matrix (ee to camera):\n", T_ee_camera)

        # T_base_aruco_v2 = T_base_ee @ delta
        # print("Original:")
        # print(T_base_aruco)
        # print("calculate")
        # print(T_base_aruco_v2)

        print("original")
        # print(T_ee_camera)
        # print("calculate")
        # T_ee_camera_v2 = delta @ np.linalg.inv(T_camera_aruco)
        # T_aruco_camera = np.linalg.inv(T_camera_aruco) 
        # print(np.linalg.inv(T_camera_aruco))
        # print("calculate")
        # T_ee_camera_v3 = delta @ T_camera_aruco
        # # print(delta[:,3] - T_aruco_camera[:,3])
        # T_base_aruco_rot = np.array([[T_base_aruco[0,0:2], 0],
        #                              [T_base_aruco[1,0:2], 0],
        #                              [T_base_aruco[2,0:2], 0],
        #                              [T_base_aruco[3,0:2], 1]])
        T_base_aruco_rot = np.array([[0.00160695 ,-0.99983118 ,-0.01830361, 0],
                                     [-0.99999777 ,-0.00158161, -0.00139917, 0],
                                     [0.00136998 , 0.01830582 ,-0.9998315, 0],
                                     [0, 0, 0, 1]])
        
        T_camera_aruco_rot = np.array([[ 0.03993667,  0.9966958,  -0.07072862,  0  ],
                                        [ 0.99712245, -0.03518888,  0.0671459,  0],
                                        [ 0.06443517, -0.07320668, -0.99523308,  0 ],
                                        [ 0,          0,         0,        1 ]])

        # T_base_aruco_rot2 = T_base_aruco
        # T_base_aruco_rot2[0,3] = 0
        # T_base_aruco_rot2[1,3] = 0
        # T_base_aruco_rot2[2,3] = 0

        # print("aruco rotation ")
        print(T_base_aruco)
        # print(T_base_aruco_rot2)

        print(" base to aruco without rotation ")
        print(T_base_aruco @ np.linalg.inv( T_base_aruco_rot  ) )

        # T_ee_cameraX = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],  [0.02233837, -0.11826378, -0.14449401, 1 ] ])
        T_ee_cameraX = np.array([[1,0,0, 0.02233837],
                                 [0,1,0,-0.11826378],
                                 [0,0,1,-0.14449401],  
                                 [0,0,0, 1 ] ])
        
          
        # print( T_base_ee @ T_ee_cameraX @ T_camera_aruco)  

        # print( T_base_aruco @ np.linalg.inv( T_base_aruco_rot) @ T_ee_cameraX @ T_camera_aruco)

        # print( T_camera_aruco @ np.linalg.inv( T_camera_aruco_rot ))  

        # print("final results")
        # print( T_base_ee @ T_ee_cameraX @ T_camera_aruco @ np.linalg.inv( T_camera_aruco_rot ) )  

        print( T_base_ee[0:2,0:2] @ T_ee_cameraX[0:2,0:2] ) 
        print( np.linalg.inv( T_camera_aruco_rot )  )   



        print( T_base_ee[:,3] +  T_ee_cameraX[:,3]  +  T_camera_aruco[:,3]  )    




        breakpoint()

        self.T_ee_camera = T_ee_camera

        



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
        
                # Detect ArUco markers
                detector = cv2.aruco.ArucoDetector(dictionary)
                markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        
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
                (trans_base_aruco, rot_base_aruco) = self.listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_aruco = self.construct_rot_matrix_homogeneous_transform(trans_base_aruco, rot_base_aruco)

        return T_base_aruco


    def calculate_delta(self, T_base_ee, T_base_aruco):
        T_base_ee_inv = np.linalg.inv(T_base_ee)
        delta = T_base_ee_inv @ T_base_aruco

        return delta

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
        # # Round small floating-point errors to zero
        # rotation_matrix = np.round(rotation_matrix, decimals=10)
        # print("rotation_matrix: ", rotation_matrix)

        return rotation_matrix

    import numpy as np

    def rotation_matrix_y(self, theta_degrees):
        """
        Generates a 3x3 rotation matrix for a rotation around the y-axis.

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

        # Construct the rotation matrix for Y-axis rotation
        rotation_matrix = np.array([
            [ cos_theta,  0, sin_theta],
            [ 0,          1, 0        ],
            [-sin_theta,  0, cos_theta]
        ])

        return rotation_matrix


    def get_base_to_camera(self):
        while not rospy.is_shutdown():
            try:
                # Get transformation from base_link to end_effector_link
                (trans_base_camera, rot_base_camera) = self.listener.lookupTransform('/base_link', '/d435_depth_optical_frame', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T_base_camera = self.construct_rot_matrix_homogeneous_transform(trans_base_camera, rot_base_camera)

        return T_base_camera
    

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
        # --- Get Base to Camera ---
        T_base_ee = self.get_base_to_ee()
        print("Transformation Matrix (base to ee):\n", T_base_ee)
        
        T_ee_camera = self.T_ee_camera
        print("Transformation Matrix (ee to camera):\n", T_ee_camera)

        T_base_camera = T_base_ee @ T_ee_camera
        print("Transformation Matrix (base to camera):\n", T_base_camera)

        # --- Get Camera to Obj ---
        trans_zero = np.zeros(3)
        # rot_z_90 = self.rotation_matrix_z(90)
        rot_y_180 = self.rotation_matrix_y(180)
        T_camera_grasp = self.construct_homogeneous_transform(trans_zero, rot_y_180)
        print("Transformation Matrix (camera to grasp):\n", T_camera_grasp)

        poses = self.read_gg_values(self.filepath)
        trans_grasp = poses["translation"]
        rot_grasp = poses["rotation"]
        print("trans: ", trans_grasp)
        print("rot: ", rot_grasp.shape)
        T_grasp_obj = self.construct_homogeneous_transform(trans_grasp, rot_grasp)
        print("Transformation Matrix (grasp to obj):\n", T_grasp_obj)
        
        T_camera_obj = T_camera_grasp @ T_grasp_obj
        print("Transformation Matrix (camera to obj):\n", T_camera_obj)
    


        # --- Get Base to Obj ---
        T_base_obj = T_base_camera @ T_camera_obj
        print("Transformation Matrix (base to obj):\n", T_base_obj)

        self.target_pos = self.transformation_to_pose(T_base_obj)
        print("Target Pose:\n", self.target_pos)

        self.gripper_width = self.read_gripper_width(self.filepath)
        print("Gripper Width:\n", self.gripper_width)

        breakpoint()


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

def main():
    rospy.init_node('grasp_planning', anonymous=True)
    grasp_planner_node = GraspPlannerNode()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()