#!/usr/bin/env python3
from lerobot_joycon.lerobot.common.robot_devices.robots.factory import make_robot
from lerobot_joycon.lerobot.common.robot_devices.robots.utils import Robot
from pathlib import Path
import rospy
import numpy as np
import torch
import pickle
import os
import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from threading import Lock
from lerobot_joycon.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot_joycon.lerobot.common.utils.utils import init_hydra_config
from functools import partial

class Recorder:
    def __init__(self,
    robot: Robot,
    root: Path
    ):
        rospy.init_node('recorder')
        self.bridge = CvBridge()
        self.lock = Lock()

        self.latest_images = {} 
        self.latest_state = None
        self.latest_action = None
        self.cameras = ['cam_1'] #TODO: 在写完xarm_config后，将这里的self.cameras的调用都改为调用 robot.cameras

        for cam_name in self.cameras:
            rospy.Subscriber(f'{cam_name}', Image, partial(self.image_callback, cam_name=cam_name))

        rospy.Subscriber('/robot_state', Float64MultiArray, self.state_callback)
        rospy.Subscriber('/robot_cmd',Float64MultiArray, self.cmd_callback)
        rospy.on_shutdown(self.save_episode)
        rospy.loginfo("Recorder initialized and listening...")

        num_image_writer_threads_per_camera = 0 # Hardcode

        self.dataset = LeRobotDataset.create( #TODO: Now is hardcode
            repo_id = 'task/so100_test',
            fps = 30,
            root= root,
            robot=robot,
            use_videos=True,
            image_writer_processes=num_image_writer_threads_per_camera,
            image_writer_threads=num_image_writer_threads_per_camera * len(robot.cameras),
        )

    def image_callback(self, msg, cam_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return

        with self.lock:
            self.latest_images[cam_name] = cv_image
            self.record_step()


    def state_callback(self, msg):
        with self.lock:
            self.latest_state = msg.data
            self.record_step()

    def cmd_callback(self, msg):
        with self.lock:
            self.latest_action = msg.data
            self.record_step()

    def record_step(self):
        if all(self.latest_images.get(name) is not None for name in self.cameras) and \
        self.latest_state is not None and \
        self.latest_action is not None:

            obs_dict = {
            "observation.state": torch.from_numpy(np.array(self.latest_state))
            }
            for name in self.cameras:
                image = self.latest_images[f'{name}']
                obs_dict[f'observation.images.{name}'] = torch.from_numpy(image.copy())
            action_dict = {
            "action": torch.from_numpy(np.array(self.latest_action)) 
            }
            self.dataset.add_frame({**obs_dict,**action_dict})
            rospy.loginfo(f"Recorded step {len(self.dataset)}")
            for cam_name in self.cameras:
                self.latest_images[cam_name] = None
            self.latest_state = None
            self.latest_action = None

    def save_episode(self):
        task = 'xarm_collection'
        self.dataset.save_episode(task)

if __name__ == '__main__':
    robot_path = 'lerobot/configs/robot/xarm.yaml '#TODO: Xarm Config
    robot_overrides = None
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    save_dir = "/root/xarm_ws/save_data"
    try:
        Recorder(robot=make_robot(robot_cfg), root="/root/xarm_ws/save_data") 
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
