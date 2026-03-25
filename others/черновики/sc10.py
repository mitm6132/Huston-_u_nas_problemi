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

        # Сервисы
        rospy.wait_for_service('navigate')
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('land')
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.land = rospy.ServiceProxy('land', Trigger)

        # Подписки
        self.detections = None
        self.range = None
        self.image = None

        self.image_width = 640
        self.image_height = 480

        rospy.Subscriber('/vision/detections', DetectionArray, self.detections_cb)
        rospy.Subscriber('/front_rangefinder/range', Range, self.range_cb)
        rospy.Subscriber('/front_main_camera/camera_info', CameraInfo, self.camera_info_cb)
        rospy.Subscriber('/front_main_camera/image_raw', Image, self.image_cb)

        self.bridge = CvBridge()

        # Параметры миссии
        self.height = 1.5
        self.target_distance = 1.5
        self.mission_duration = 120.0

        # Управление
        self.kp_yaw = 0.002
        self.kp_dist = 0.8
        self.max_speed_x = 0.3
        self.center_threshold = 50

        # Состояния
        self.state = 'SEARCH'
        self.selected_target = None
        self.lost_start = None
        self.search_direction = 1

        # Таймеры
        self.last_cmd_time = 0
        self.cmd_interval = 0.2

    # ---------------- CALLBACKS ----------------

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def camera_info_cb(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

    def image_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            pass

    # ---------------- UTILS ----------------

    def get_black_center(self, target):
        if self.image is None:
            return None

        x1 = int(target.x_min)
        x2 = int(target.x_max)
        y1 = int(target.y_min)
        y2 = int(target.y_max)

        roi = self.image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        if cv2.countNonZero(mask) < 30:
            return None

        M = cv2.moments(mask)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])

        return x1 + cx

    def limit_yaw(self, yaw):
        max_yaw = math.radians(20)
        return max(-max_yaw, min(max_yaw, yaw))

    def select_target(self):
        if self.detections is None or not self.detections.detections:
            return None

        persons = [d for d in self.detections.detections if d.class_name == "person"]

        if not persons:
            return None

        return max(persons, key=lambda d: (d.x_max - d.x_min)*(d.y_max - d.y_min))

    def get_center(self, target):
        return (target.x_min + target.x_max) // 2

    def send_cmd(self, move_x, yaw):
        telem = self.get_telemetry(frame_id='aruco_map')
        err_z = self.height - telem.z
        move_z = max(-0.2, min(0.2, 0.8 * err_z))

        now = time.time()

        if now - self.last_cmd_time > self.cmd_interval:
            self.navigate(x=move_x,
                          y=0.0,
                          z=move_z,
                          yaw=yaw,
                          speed=0.5,
                          frame_id='body')
            self.last_cmd_time = now

    # ---------------- MAIN ----------------

    def run(self):

        rospy.loginfo("Взлёт")
        self.navigate(z=self.height, frame_id='body', auto_arm=True)
        rospy.sleep(5)

        start = time.time()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            if time.time() - start > self.mission_duration:
                break

            target = self.select_target()

            if target is None:
                rospy.loginfo_throttle(2, "Поиск цели...")
                self.send_cmd(0.0, 0.05)  # вращение
                rate.sleep()
                continue

            # --- выбор центра (черный или bbox) ---
            black_cx = self.get_black_center(target)

            if black_cx is not None:
                cx = black_cx
            else:
                cx = self.get_center(target)

            err_x = cx - self.image_width / 2

            # --- yaw управление ---
            yaw = -self.kp_yaw * err_x
            yaw = self.limit_yaw(yaw)

            # --- движение вперед ---
            move_x = 0.0
            if abs(err_x) < self.center_threshold and self.range is not None:
                err_dist = self.range - self.target_distance
                move_x = self.kp_dist * err_dist
                move_x = max(-self.max_speed_x, min(self.max_speed_x, move_x))

            self.send_cmd(move_x, yaw)

            rospy.loginfo_throttle(1,
                "dist=%.2f err=%.0f yaw=%.2f move_x=%.2f",
                self.range if self.range else 0, err_x, yaw, move_x)

            rate.sleep()

        rospy.loginfo("Посадка")
        self.land()


if __name__ == '__main__':
    try:
        TargetTracker().run()
    except rospy.ROSInterruptException:
        pass