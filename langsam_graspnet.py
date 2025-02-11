#!/usr/bin/env python3

import cv2
import requests
import numpy as np
import threading
 
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
# from playsound import playsound
import os
import base64
 
import tf
import sys
import rospy
import moveit_commander
from control_msgs.msg import GripperCommandActionGoal
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String  # Import the String message for communication
import os

import pyrealsense2 as rs
import requests
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import cv2
import os
import base64

import torch
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

import open3d as o3d  # 3D data processing and visualization
import argparse  # Command-line argument parsing
import scipy.io as scio  # Handling MATLAB files

import argparse
import sys

# sys.argv = [
#     '--checkpoint_path', '/path/to/checkpoint',
# ]
 
class LangsamGraspnetNode():
    def __init__(self):
        # Initialize parameters
        self.init_params()

        # Initialize and parse the arguments here instead of terminal
        parser = self.initialize_parser()
        # Use the arguments programmatically by passing a pre-defined list to parse_args
        args = ['--checkpoint_path', '/home/koyo_ws/graspnet/graspnet-baseline/checkpoint-rs.tar']
        self.cfgs = parser.parse_args(args)
 
        # Object's position
        self.Obj_pose = PoseStamped()
        self.Obj_pose.pose.position.x = 0

        self.done_graspnet = False
 
 

    def init_params(self):
        # self.server_url = "http://127.0.0.1:8000/predict"
        self.server_url = "http://172.22.247.138:8000/predict" 

        self.record_duration = 5
        self.audio_file_name = "audio.mp3"
        self.audio_file_save_dir = "audio_files"

        self.whisper_model = "base"

        self.image_w = 1280.0
        self.image_h = 720.0
        self.frame_rate = 30
        self.image_file_save_dir = "image_files"

        self.sam_type = "sam2.1_hiera_small"
        self.box_threshold = 0.3
        self.text_threshold = 0.25

    def initialize_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
        parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
        parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
        parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
        parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
        return parser

    def start_demo(self):
        print("----------------------------------------------------")
        print("DEMO START")
        self.start_langsam() # langsam, whispe
        self.demo() # graspnet
    

    ####### langsam, whisper #######

    def record_audio(self, duration, filename, sample_rate=44100):
        save_dir = self.audio_file_save_dir
        os.makedirs(save_dir, exist_ok=True)
        wav_filename = os.path.join(save_dir, "temp.wav")
        mp3_filename = os.path.join(save_dir, filename)
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        write(wav_filename, sample_rate, audio_data)
        sound = AudioSegment.from_wav(wav_filename)
        sound.export(mp3_filename, format="mp3")
        print(f"Recording saved as {mp3_filename}")
 
    def start_langsam(self):

        # self.record_audio(duration=self.record_duration, filename=self.audio_file_name)

        # selected_model = self.whisper_model
        # model = whisper.load_model(selected_model)
        # print("----------------------------------------------------")
        # print("Processing speech-to-text")
        # result = model.transcribe("audio_files/audio.mp3")
        # print(result["text"])
        # print("----------------------------------------------------")

        # RealSense camera setup
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.image_w, self.image_h, rs.format.bgr8, self.frame_rate)
        config.enable_stream(rs.stream.depth, self.image_w, self.image_h, rs.format.z16, self.frame_rate)

        try:
            pipeline.start(config)
            print("Streaming started. Press 'q' to quit.")
            save_dir = self.image_file_save_dir
            os.makedirs(save_dir, exist_ok=True)
            count = 0
            color_image = None
            depth_image = None

            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    print("Error: Could not read frame.")
                    continue

                # Convert RealSense frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Normalize depth image for display
                depth_image_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

                # Display images
                cv2.imshow("Color Image", color_image)
                cv2.imshow("Depth Image", depth_image_display)

                count += 1

                # Convert the color frame to PNG format
                _, buffer = cv2.imencode('.png', color_image)

                # Prepare the payload
                files = {
                    "sam_type": (None, self.sam_type),
                    "box_threshold": (None, self.box_threshold),
                    "text_threshold": (None, self.text_threshold),
                    "text_prompt": (None, result["text"]),
                    "image": ("image.png", buffer.tobytes(), "image/png")
                }

                # Send the POST request
                response = requests.post(self.server_url, files=files)

                if response.status_code == 200:
                    response_json = response.json()
                    print(f"Processed output received for frame {count}")

                    # Decode the base64-encoded image
                    img_data = base64.b64decode(response_json["output_image"])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    output_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    # Display the processed output image
                    # cv2.imshow("Output Image", output_image)

                    # Get the masks
                    masks = response_json["masks"]
                    masks = masks[0]
                    masks = np.array(masks)
                    mask_image = (masks * 255).astype(np.uint8)  # Scale 0 to 1 values to 0 to 255
                    # Display the mask image
                    cv2.imshow("Mask Image", mask_image)

                    # Get the boxes
                    boxes = response_json["boxes"]
                    print("boxes: ", boxes)

                    # Get the labels
                    labels = response_json["labels"]
                    print("labels: ", labels)

                    # Get the scores
                    scores = response_json["scores"]
                    print("scores: ", scores)

                    # Get the num_object
                    num_object = response_json["num_object"]
                    print("number of object: ", num_object)

                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if count == 10:
                    print("Break the loop")
                    break

        finally:
            pipeline.stop()

            # Save the last captured color and depth images
            color_image_path = os.path.join(save_dir, "color_image.png")
            depth_image_path = os.path.join(save_dir, "depth_image.png")
            output_image_path = os.path.join(save_dir, "output_image.png")
            mask_image_path = os.path.join(save_dir, "mask_image.png")

            cv2.imwrite(color_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(output_image_path, output_image)
            cv2.imwrite(mask_image_path, mask_image)

            print(f"Color image saved as {color_image_path}")
            print(f"Depth image saved as {depth_image_path}")
            print(f"Output image saved as {output_image_path}")
            print(f"Mask image saved as {mask_image_path}")

            cv2.destroyAllWindows()

            self.color_image = color_image
            self.depth_image = depth_image
            self.mask_image = mask_image
            if num_object == 1:
                box = boxes[0]
                self.xmin, self.ymin, self.xmax, self.ymax = box[0], box[1], box[2], box[3]
            else:
                raise ValueError("More than two obejects are detected !!")
            
    ####### graspnet #######
 
    def demo(self, metadata_dir):
        net = self.get_net()
        end_points, cloud = self.get_and_process_data(metadata_dir)
        gg = self.get_grasps(net, end_points)
        if self.cfgs.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        self.vis_grasps(gg, cloud)
        self.connect_to_planner()

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.cfgs.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self, metadata_dir):
        # load data
        # color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        color = self.color_image
        depth = self.depth_image
        workspace_mask = self.mask_image

        meta = scio.loadmat(os.path.join(metadata_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # Manually update intrinsic parameters (overriding values from metadata).
        intrinsic[0][0] = 636.4911
        intrinsic[1][1] = 636.4911
        intrinsic[0][2] = 642.3791
        intrinsic[1][2] = 357.4644
        factor_depth = 1000  # Set depth scaling factor for the camera.

        # generate cloud
        camera = CameraInfo(self.image_w, self.image_h, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg
    
    def vis_grasps(self, gg, cloud):
        if len(gg) == 0:
            raise ValueError("No valid grasps exist !!")

        gg.nms()
        gg.sort_by_score()
        top_grasp = gg[0]  # Select the highest-scoring grasp.
        print("top grasp: ", top_grasp)
        # top_grasp_str = str(top_grasp)

        self.gg = top_grasp

        # # Append the top grasp to a log file for record-keeping.
        # gg_file_path = '/home/chart-admin/catkin_workspace/data/gg_values.txt'
        # with open(gg_file_path, 'a') as file:
        #     file.write(f"Object Name: {object_name}\n")
        #     file.write("Top Grasp Value:\n")
        #     file.write(top_grasp_str)
        #     file.write("\n\n")

        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])


    def connect_to_planner(self):
        self.translation = self.gg
        self.rotation_matrix = self.gg
        self.done_graspnet = True

    ####### broadcaster to make transformation between base_link and /grasp #######
    
    def broadcast_static_tf(self):
        if not self.done_graspnet:
            return

        broadcaster = tf.TransformBroadcaster()
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # Using the translation values read from the file
            translation = tuple(translation)

            # Using the rotation matrix values read from the file
            rotation_matrix = np.array(rotation_matrix).reshape((3, 3))

            # Convert the rotation matrix to a quaternion
            quaternion_original = tf.transformations.quaternion_from_matrix(
                np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
            )

            quaternion_x180 = tf.transformations.quaternion_about_axis(np.pi, (1, 0, 0))
            combined_quaternion_x = tf.transformations.quaternion_multiply(quaternion_original, quaternion_x180)

            # Quaternion for a 90 degree rotation around the y-axis
            quaternion_y90 = tf.transformations.quaternion_about_axis(np.pi / 2, (0, 1, 0))

            # Quaternion for a 90 degree rotation around the z-axis
            quaternion_z90 = tf.transformations.quaternion_about_axis(np.pi / 2, (0, 0, 1))

            # Combine the quaternions: first apply the rotation around y, then around z
            combined_quaternion_y = tf.transformations.quaternion_multiply(quaternion_original, quaternion_y90)
            combined_quaternion_z = tf.transformations.quaternion_multiply(combined_quaternion_y, quaternion_z90)

            # Broadcasting the transform
            # transformation from camera to grasp
            # grasp frame is located at a specific position and orientation relative to the depth camera's optical frame.
            broadcaster.sendTransform(
                translation, combined_quaternion_z, rospy.Time.now(), 'grasp', 'd435_depth_optical_frame'
            )



def main():
    rospy.init_node('langsam_graspnet', anonymous=True)
    rospy.loginfo('Start Langsam and Graspnet')
    langsam_graspnet_node = LangsamGraspnetNode()

    # Start the broadcaster in a separate thread after graspnet
    rospy.loginfo('Starting listener for grasp planning ...')
    broadcaster_thread = threading.Thread(target=langsam_graspnet_node.broadcast_static_tf)
    broadcaster_thread.start()

    langsam_graspnet_node.start_demo()
    rospy.spin()
 
if __name__ == "__main__":
    main()