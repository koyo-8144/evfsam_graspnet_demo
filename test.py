import cv2
import numpy as np
import pyrealsense2 as rs


def construct_homogeneous_transform(translation, rotation_matrix):
    # Construct the 4x4 homogeneous transformation matrix
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T
 
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
markerLength = 0.066  # 6.6 cm
 
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

                T_camra_arco = construct_homogeneous_transform(tvec, rotation_matrix)
                print("Transformation Matrix (camera to arco):\n", T_camra_arco)

            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
 
        # Show frame
        cv2.imshow("RealSense ArUco Pose Estimation", frame)
 
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
 
 
 
 
 
# # Taking a 3 * 3 matrix
# A = np.array([[6, 1, 1],
#           [4, -2, 5],
#           [2, 8, 7]])
 
# # Calculating the inverse of the matrix
# print(np.linalg.inv(A))
 
 
 
# # Taking a 4 * 4 matrix
# A = np.array([[6, 1, 1, 3],
#           [4, -2, 5, 1],
#           [2, 8, 7, 6],
#           [3, 1, 9, 7]])
 
# # Calculating the inverse of the matrix
# print(np.linalg.inv(A))
 