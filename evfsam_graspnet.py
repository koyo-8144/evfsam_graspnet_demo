import cv2
import requests
import numpy as np

import os
import base64
 
import sys
import os
from PIL import Image
 
import pyrealsense2 as rs
from scipy.io.wavfile import write
import cv2
import os
import base64
 
import torch
import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoTokenizer, BitsAndBytesConfig
 
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
 
import sys
sys.path.append('/home/sandisk/koyo_ws/graspnet-baseline/models')
# sys.path.append('/home/koyo/graspnet-baseline/models')
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
 
sys.path.append("/home/sandisk/koyo_ws/graspnetAPI")
# sys.path.append("/home/koyo/graspnetAPI")
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
 
sys.path.append("/home/sandisk/koyo_ws/graspnet-baseline/dataset")
# sys.path.append("/home/koyo/graspnet-baseline/dataset")
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
 
import open3d as o3d  # 3D data processing and visualization
import argparse  # Command-line argument parsing
import scipy.io as scio  # Handling MATLAB files
 
import argparse

# import paho.mqtt.client as mqtt

# Define the broker address and port
BROKER_ADDRESS = "172.22.247.109"
BROKER_PORT = 15672

# Define the server URL
url = "http://100.106.58.3:8000/predict"  # Note the '/predict' endpoint


class EVFsamGraspnet():
    def __init__(self):
        # Initialize parameters
        self.init_params()

        # self.tokenizer, self.model = self.init_models()
        
        # self.client = mqtt.Client()
        # self.client.on_connect = self.on_connect
        # self.client.on_message = self.on_message
 
    def init_params(self):
        # Graspnet params
        self.checkpoint_path = "/home/sandisk/koyo_ws/graspnet-baseline/checkpoint-rs.tar"
        self.num_point = 20000
        self.num_view = 300
        self.collision_thresh = 0.01
        self.voxel_size = 0.01

        # # EVF-SAM params
        # self.version = "YxZhang/evf-sam2"
        # self.vis_save_path = "./infer"
        # self.precision = "fp16" # "fp32", "bf16", "fp16"
        # self.image_size = 224
        # self.model_max_length = 512
        # self.local_rank = 0
        # self.load_in_8bit = False
        # self.load_in_4bit = False
        # self.model_type = "ori" # "ori", "effi", "sam2"
        # self.image_path = "assets/zebra.jpg"
        self.prompt = "pick yellow pole"
        # self.prompt = "pick up a blue cup"

        self.image_w = 1280
        self.image_h = 720
        self.frame_rate = 30
        self.image_file_save_dir = "image_files"
 
        self.display_count = 0
        self.display_itr = 5
 
        self.gg_file_path = '/home/sandisk/koyo_ws/demo_ws/src/demo_pkg/evfsam_graspnet_demo/data/gg_values.txt'
 
        self.data_path = '/home/sandisk/koyo_ws/demo_ws/src/demo_pkg/evfsam_graspnet_demo/data'
 

    def start_demo(self):
        print("----------------------------------------------------")
        print("DEMO START")
        self.start_evfsam() # langsam, whispe
        self.demo(self.data_path) # graspnet
    
    ####### MQTT Setup #######

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to Mosquitto broker!")
            self.client.subscribe("robot/arrival")
            print("Waiting for message .....")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        print(f"Message received: {msg.payload.decode()}")
        self.object = msg.payload.decode()
        print("Received object name: ", self.object)
        client.loop_stop()
 
    ####### EVF-SAM Client #######
 
    def start_evfsam(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        try:
            # pipeline.start(config)
            # Start streaming
            pipeline_profile = pipeline.start(config)
            # Get camera intrinsics
            profile = pipeline_profile.get_stream(rs.stream.color)
            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            camMatrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]])
            distCoeffs = np.zeros(5)  # Assuming no lens distortion

            print("Streaming started. Press 'q' to quit.")
            save_dir = "image_files"
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

                # Flip the images horizontally (left-to-right) and vertically (up-to-down)
                color_image = cv2.flip(color_image, -1)  # Flip both horizontally and vertically
                depth_image = cv2.flip(depth_image, -1)  # Flip both horizontally and vertically

                # Normalize depth image for display
                depth_display_image = cv2.convertScaleAbs(depth_image, alpha=0.03)

                # Send the POST requestDisplay images
                cv2.imshow("Color Image", color_image)
                cv2.imshow("Depth Image", depth_display_image)

                # Convert the color frame to PNG format
                _, buffer = cv2.imencode('.png', color_image)

                count += 1

                # Prepare the payload
                files = {
                    "text_prompt": (None, self.prompt),
                    "image": ("image.png", buffer.tobytes(), "image/png")
                }

                # print("Send the POST request")
                # Send the POST request
                response = requests.post(url, files=files)

                if response.status_code == 200:
                    response_json = response.json()
                    print(f"Processed output received for frame {count}")

                    # Decode the base64-encoded image
                    seg_img_data = base64.b64decode(response_json["segmentation_image"])
                    seg_img_array = np.frombuffer(seg_img_data, dtype=np.uint8)
                    segmentation_image = cv2.imdecode(seg_img_array, cv2.IMREAD_COLOR)
                    # Display the processed output image
                    cv2.imshow("Segmentation Image", segmentation_image)

                    # Decode the base64-encoded image
                    bb_img_data = base64.b64decode(response_json["bounding_box_image"])
                    bb_img_array = np.frombuffer(bb_img_data, dtype=np.uint8)
                    bounding_box_image = cv2.imdecode(bb_img_array, cv2.IMREAD_COLOR)
                    # Display the processed output image
                    cv2.imshow("Bounding Box Image", bounding_box_image)

                    # Decode the base64-encoded image
                    mask_img_data = base64.b64decode(response_json["mask_image"])
                    mask_img_array = np.frombuffer(mask_img_data, dtype=np.uint8)
                    # img_array = (img_array * 255).astype(np.uint8)
                    mask_image = cv2.imdecode(mask_img_array, cv2.IMREAD_COLOR)
                    # mask_image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    # print("mask_image: ", mask_image)
                    # Display the processed output image
                    cv2.imshow("Mask Image", mask_image)

                    xmin = response_json["xmin"]
                    ymin = response_json["ymin"]
                    xmax = response_json["xmax"]
                    ymax = response_json["ymax"]
                    print("xmin: ", xmin)
                    print("ymin: ", ymin)
                    print("xmax: ", xmax)
                    print("ymax: ", ymax)

                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.display_count += 1
                if self.display_count == self.display_itr:
                    print("Break the loop")
                    break

        finally:
            pipeline.stop()

            # Save the last captured color and depth images
            color_image_path = os.path.join(save_dir, "color_image.png")
            depth_image_path = os.path.join(save_dir, "depth_image.png")
            depth_display_image_path = os.path.join(save_dir, "depth_display_image.png")
            segmentation_image_path = os.path.join(save_dir, "segmentation_image.png")
            bounding_box_image_path = os.path.join(save_dir, "bounding_box_image.png")
            mask_image_path = os.path.join(save_dir, "mask_image.png")

            cv2.imwrite(color_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(depth_display_image_path, depth_display_image)
            cv2.imwrite(segmentation_image_path, segmentation_image)
            cv2.imwrite(bounding_box_image_path, bounding_box_image)
            cv2.imwrite(mask_image_path, mask_image)

            print(f"Color image saved as {color_image_path}")
            print(f"Depth image saved as {depth_image_path}")
            print(f"Depth display image saved as {depth_display_image_path}")
            print(f"Segmentation image saved as {segmentation_image_path}")
            print(f"Bounding box image saved as {bounding_box_image_path}")
            print(f"Mask image saved as {mask_image_path}")

            self.color_image = color_image
            self.depth_image = depth_image
            self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
            self.intrinsics = intrinsics

            cv2.destroyAllWindows()
    
            
    ####### graspnet #######
 
    def demo(self, data_dir):
        # Clear previous grasp results at the start of the demo.
        gg_file_path = self.gg_file_path
        with open(gg_file_path, 'w') as file:
            file.write("")
 
        net = self.get_net()
        # end_points, cloud = self.get_and_process_data(data_dir)
        # end_points, cloud, depth, intrinsic, factor_depth = self.get_and_process_data(data_dir)
        end_points, cloud, depth, intrinsic, factor_depth = self.get_filter_process_data(data_dir)
        
        # # Extract bounding box coordinates in 3D.
        # xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax
        # xmin_3d_x, xmin_3d_y = self.pixel_to_camera_coords(xmin, ymin, depth, intrinsic, factor_depth)
        # xmax_3d_x, xmax_3d_y = self.pixel_to_camera_coords(xmax, ymax, depth, intrinsic, factor_depth)
        # print("xmin_3d_x: ", xmin_3d_x)
        # print("xmin_3d_y: ", xmin_3d_y)
        # print("xmax_3d_x: ", xmax_3d_x)
        # print("xmax_3d_y: ", xmax_3d_y)

        # Extract 2D bounding box coordinates
        xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax

        print("Top-left in camera: ", (xmin, ymin))
        print("Top-right in camera: ", (xmax, ymin))
        print("Bottom-left in camera: ", (xmin, ymax))
        print("Bottom-right in camera: ", (xmax, ymax))
        # print("xmin: ", xmin)
        # print("ymin: ", ymin)
        # print("xmax: ", xmax)
        # print("ymax: ", ymax)

        # Convert all 4 corners to 3D coordinates
        xmin_ymin_3d = self.pixel_to_camera_coords(xmin, ymin, depth, intrinsic, factor_depth)  # Top-left
        xmax_ymin_3d = self.pixel_to_camera_coords(xmax, ymin, depth, intrinsic, factor_depth)  # Top-right
        xmin_ymax_3d = self.pixel_to_camera_coords(xmin, ymax, depth, intrinsic, factor_depth)  # Bottom-left
        xmax_ymax_3d = self.pixel_to_camera_coords(xmax, ymax, depth, intrinsic, factor_depth)  # Bottom-right

        print("Top-left in 3d: ", xmin_ymin_3d)
        print("Top-right in 3d: ", xmax_ymin_3d)
        print("Bottom-left in 3d: ", xmin_ymax_3d)
        print("Bottom-right in 3d: ", xmax_ymax_3d)
        # breakpoint()

        # Generate and filter grasps.
        gg = self.get_grasps(net, end_points)
        print("Original gg: ", gg)
        filtered_gg = GraspGroup()

        # for grasp in gg:
        #     translation = grasp.translation
        #     # if (xmin_3d_x - 0.1  <= translation[0] <= xmax_3d_x + 0.1 and
        #     #     xmin_3d_y - 0.1 <= translation[1] <= xmax_3d_y + 0.1):
        #     if (xmin_3d_x  <= translation[0] <= xmax_3d_x  and
        #         xmin_3d_y  <= translation[1] <= xmax_3d_y ):
        #         filtered_gg.add(grasp)

        # # Check if grasp points are within the 3D bounding box
        # for grasp in gg:
        #     translation = grasp.translation
        #     if (min(xmin_ymin_3d[0], xmax_ymin_3d[0], xmin_ymax_3d[0], xmax_ymax_3d[0]) <= translation[0] <= max(xmin_ymin_3d[0], xmax_ymin_3d[0], xmin_ymax_3d[0], xmax_ymax_3d[0]) and
        #         min(xmin_ymin_3d[1], xmax_ymin_3d[1], xmin_ymax_3d[1], xmax_ymax_3d[1]) <= translation[1] <= max(xmin_ymin_3d[1], xmax_ymin_3d[1], xmin_ymax_3d[1], xmax_ymax_3d[1])):
        #         filtered_gg.add(grasp)

        margin = 0.05  # Define a margin of 2 cm

        min_x = min(xmin_ymin_3d[0], xmax_ymin_3d[0], xmin_ymax_3d[0], xmax_ymax_3d[0]) - margin
        max_x = max(xmin_ymin_3d[0], xmax_ymin_3d[0], xmin_ymax_3d[0], xmax_ymax_3d[0]) + margin
        min_y = min(xmin_ymin_3d[1], xmax_ymin_3d[1], xmin_ymax_3d[1], xmax_ymax_3d[1]) - margin
        max_y = max(xmin_ymin_3d[1], xmax_ymin_3d[1], xmin_ymax_3d[1], xmax_ymax_3d[1]) + margin

        for grasp in gg:
            translation = grasp.translation
            if (min_x <= translation[0] <= max_x and min_y <= translation[1] <= max_y):
                filtered_gg.add(grasp)


        # Perform collision detection if needed.
        if self.collision_thresh > 0:
            filtered_gg = self.collision_detection(filtered_gg, np.array(cloud.points))

        print("Filtered gg: ", filtered_gg)
        # Log the top grasp for the object.
        self.vis_grasps(filtered_gg, cloud)
        # self.vis_grasps(gg, cloud)

        # gg = self.get_grasps(net, end_points)
        # if self.collision_thresh > 0:
        #     gg = self.collision_detection(gg, np.array(cloud.points))
        # self.vis_grasps(gg, cloud)
        
    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
 
    def get_and_process_data(self, data_dir):
        # load data
        # color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        color = self.color_image
        depth = self.depth_image
        # workspace_mask = self.mask_image
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        # print("defo mast ", workspace_mask.shape) # (720, 1280)
        # breakpoint()
 
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
 
        # # Manually update intrinsic parameters (overriding values from metadata).
        # intrinsic[0][0] = 636.4911
        # intrinsic[1][1] = 636.4911
        # intrinsic[0][2] = 642.3791
        # intrinsic[1][2] = 357.4644
        # factor_depth = 1000  # Set depth scaling factor for the camera.

        # print(self.intrinsics.fx)
        # print(self.intrinsics.fy)
        # print(self.intrinsics.ppx)
        # print(self.intrinsics.ppy)
        intrinsic[0][0] = self.intrinsics.fx
        intrinsic[1][1] = self.intrinsics.fy
        intrinsic[0][2] = self.intrinsics.ppx
        intrinsic[1][2] = self.intrinsics.ppy
        factor_depth = 1000  # Set depth scaling factor for the camera
        # breakpoint()
 
        # generate cloud
        camera = CameraInfo(self.image_w, self.image_h, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
 
        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]
 
        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
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
 
        # return end_points, cloud
        return end_points, cloud, depth, intrinsic, factor_depth

    def get_filter_process_data(self, data_dir):
        # Load data
        color = self.color_image
        depth = self.depth_image
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))

        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # Update intrinsic parameters
        intrinsic[0][0] = self.intrinsics.fx
        intrinsic[1][1] = self.intrinsics.fy
        intrinsic[0][2] = self.intrinsics.ppx
        intrinsic[1][2] = self.intrinsics.ppy
        factor_depth = 1000  # Ensure depth scaling factor is correct

        # Apply depth threshold to remove noise
        min_depth = 0.2  # Minimum depth (20 cm)
        max_depth = 0.8  # Maximum depth (1.5 m)
        depth[depth < min_depth * factor_depth] = 0
        depth[depth > max_depth * factor_depth] = 0

        # Generate point cloud
        camera = CameraInfo(self.image_w, self.image_h, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # Apply workspace mask
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # Convert masked point cloud to Open3D format
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        # Apply pass-through filter (limit to a specific region of interest)
        min_x, max_x = -0.5, 0.5
        min_y, max_y = -0.5, 0.5
        min_z, max_z = min_depth, max_depth
        filtered_points = []
        for point in np.asarray(cloud_o3d.points):
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y and min_z <= point[2] <= max_z:
                filtered_points.append(point)
        cloud_o3d.points = o3d.utility.Vector3dVector(np.array(filtered_points))

        # # Apply voxel grid filter for downsampling
        # voxel_size = 0.01  # 1 cm voxel grid
        # cloud_o3d = cloud_o3d.voxel_down_sample(voxel_size=voxel_size)

        # # Remove statistical outliers to clean up the point cloud
        # cl, ind = cloud_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # cloud_o3d = cloud_o3d.select_by_index(ind)

        # # Optional: Visualize the filtered point cloud
        # o3d.visualization.draw_geometries([cloud_o3d], window_name="Filtered Point Cloud")

        # Sample points from the filtered point cloud
        cloud_masked = np.asarray(cloud_o3d.points)
        color_masked = np.asarray(cloud_o3d.colors)

        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # Convert data to tensors
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled


        # Return processed data
        return end_points, cloud_o3d, depth, intrinsic, factor_depth

    
    # def pixel_to_camera_coords(self, x, y, depth, intrinsic, factor_depth):
    #     Z = depth[y, x] / factor_depth  # Compute the depth value
    #     X = (x - intrinsic[0][2]) * Z / intrinsic[0][0]  # Compute X in camera coordinates
    #     Y = (y - intrinsic[1][2]) * Z / intrinsic[1][1]  # Compute Y in camera coordinates
    #     return X, Y, Z  # Include Z in the return statement

    def pixel_to_camera_coords(self, x, y, depth, intrinsic, factor_depth):
        Z = depth[y, x] / factor_depth  # Depth (distance from camera)
        X = (x - intrinsic[0][2]) * Z / intrinsic[0][0]  # X in camera coordinates
        Y = (y - intrinsic[1][2]) * Z / intrinsic[1][1] 
        # Y = -(y - intrinsic[1][2]) * Z / intrinsic[1][1]  # Invert Y to increase upwards
        return X, Y, Z



    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
 
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
 
    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg
    
    def vis_grasps(self, gg, cloud):
        if len(gg) == 0:
            raise ValueError("No valid grasps exist !!")
 
        gg.nms()
        gg.sort_by_score()
        top_grasp = gg[0]  # Select the highest-scoring grasp.
        print("top grasp: ", top_grasp)
        top_grasp_str = str(top_grasp)
 
        prompt_str = str(self.prompt)
 
        # self.gg = top_grasp
 
        # Append the top grasp to a log file for record-keeping.
        gg_file_path = self.gg_file_path
        with open(gg_file_path, 'a') as file:
            file.write(f"Prompt: {prompt_str}\n")
            file.write("Top Grasp Value:\n")
            file.write(top_grasp_str)
            file.write("\n\n")
 
        print("Point cloud info:")
        print(f"Number of points: {len(np.asarray(cloud.points))}")
        print(f"Number of colors: {len(np.asarray(cloud.colors))}")
 
        
        cloud.paint_uniform_color([1, 0, 0])  # Paint the point cloud red
 
 
        gg = gg[:1]
        print("gg: ", gg)
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
 

 
 
def main():
    evfsam_graspnet = EVFsamGraspnet()
    evfsam_graspnet.start_demo()
 
if __name__ == "__main__":
    main()