#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
import cv2
import numpy as np

from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range, CameraInfo, Image
from clover_yolo.msg import DetectionArray
from cv_bridge import CvBridge


class TargetTracker:
    def __init__(self):
        rospy.init_node('target_tracker', anonymous=True)

        rospy.wait_for_service('navigate')
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('land')

        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.land = rospy.ServiceProxy('land', Trigger)

        self.detections = None
        self.range = None
        self.image = None

        self.image_width = 640

        rospy.Subscriber('/vision/detections', DetectionArray, self.detections_cb)
        rospy.Subscriber('/front_rangefinder/range', Range, self.range_cb)
        rospy.Subscriber('/front_main_camera/camera_info', CameraInfo, self.camera_info_cb)
        rospy.Subscriber('/front_main_camera/image_raw', Image, self.image_cb)

        self.bridge = CvBridge()

        self.height = 1.5

        self.kp_yaw = 0.002

    # ---------- CALLBACKS ----------

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def camera_info_cb(self, msg):
        self.image_width = msg.width

    def image_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            pass

    # ---------- UTILS ----------

    def select_target(self):
        if not self.detections:
            return None

        persons = [d for d in self.detections.detections if d.class_name == "person"]
        if not persons:
            return None

        return max(persons, key=lambda d: (d.x_max - d.x_min))

    def get_center(self, target):
        return (target.x_min + target.x_max) / 2

    # ---------- MAIN ----------

    def run(self):

        rospy.loginfo("Взлёт")
        self.navigate(z=self.height, frame_id='body', auto_arm=True)
        rospy.sleep(5)

        while not rospy.is_shutdown():

            target = self.select_target()

            # -------- НЕТ ЦЕЛИ --------
            if target is None:
                rospy.loginfo("Нет цели → крутимся")

                self.navigate(yaw=0.1, frame_id='body')
                rospy.sleep(1)
                continue

            # -------- ЕСТЬ ЦЕЛЬ --------

            cx = self.get_center(target)
            err_x = cx - self.image_width / 2

            yaw = -self.kp_yaw * err_x

            # ограничение ±20°
            yaw = max(-math.radians(20), min(math.radians(20), yaw))

            rospy.loginfo("Цель есть → поворот и шаг вперед")

            # 1. повернуться
            self.navigate(yaw=yaw, frame_id='body')
            rospy.sleep(1)

            # 2. шаг вперед 1 метр
            self.navigate(x=1.0, frame_id='body')
            rospy.sleep(1)

        rospy.loginfo("Посадка")
        self.land()


if __name__ == '__main__':
    try:
        TargetTracker().run()
    except rospy.ROSInterruptException:
        pass