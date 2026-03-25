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
        rospy.init_node('tracker')

        rospy.wait_for_service('navigate')
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('land')

        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.land = rospy.ServiceProxy('land', Trigger)

        # topics
        self.detections = None
        self.range = None
        self.image = None

        rospy.Subscriber('/front_vision/detections', DetectionArray, self.detections_cb)
        rospy.Subscriber('/front_rangefinder/range', Range, self.range_cb)
        rospy.Subscriber('/front_main_camera/camera_info', CameraInfo, self.cam_cb)
        rospy.Subscriber('/front_main_camera/image_raw', Image, self.image_cb)

        self.bridge = CvBridge()

        # параметры
        self.height = 1.5
        self.target_distance = 1.0

        self.image_width = 640

        self.state = 'SEARCH'

        self.kp_yaw = 0.005
        self.kp_dist = 0.4

        self.last_cmd = 0

    # ---------------- callbacks ----------------

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def cam_cb(self, msg):
        self.image_width = msg.width

    def image_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            pass

    # ---------------- utils ----------------

    def detect_green(self):
        if self.image is None:
            return False

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv,
                           np.array([40, 60, 60]),
                           np.array([80, 255, 255]))

        ratio = cv2.countNonZero(mask) / mask.size

        return ratio > 0.05

    def get_person(self):
        if not self.detections:
            return None

        persons = [d for d in self.detections.detections if d.class_name == "person"]

        if not persons:
            return None

        return max(persons, key=lambda d: (d.x_max - d.x_min))

    def get_center(self, t):
        return (t.x_min + t.x_max) / 2

    def send_cmd(self, x=0, y=0, z=0, yaw=float('nan')):
        if time.time() - self.last_cmd > 0.2:
            self.navigate(x=x, y=y, z=z,
                          yaw=yaw,
                          frame_id='body',
                          speed=0.5)
            self.last_cmd = time.time()

    # ---------------- MAIN ----------------

    def run(self):

        rospy.loginfo("Взлёт")
        self.navigate(z=self.height, frame_id='body', auto_arm=True)
        rospy.sleep(5)

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            # --- AVOID ---
            if self.detect_green():
                rospy.loginfo("Облет сверху")

                # вверх
                self.navigate(z=2.0, frame_id='aruco_map')
                rospy.sleep(3)

                # вперед
                self.navigate(x=1.0, frame_id='body')
                rospy.sleep(3)

                # вниз
                self.navigate(z=1.5, frame_id='aruco_map')
                rospy.sleep(3)

                continue

            # --- TRACK ---
            person = self.get_person()

            if person:
                cx = self.get_center(person)

                err = cx - self.image_width / 2

                # yaw управление
                yaw = - self.kp_yaw * err

                move_x = 0

                if self.range:
                    dist_err = self.range - self.target_distance
                    move_x = self.kp_dist * dist_err

                self.send_cmd(x=move_x, yaw=yaw)

            else:
                # поиск (вращение)
                self.send_cmd(yaw=0.3)

            rate.sleep()

        self.land()


if __name__ == '__main__':
    try:
        TargetTracker().run()
    except rospy.ROSInterruptException:
        pass