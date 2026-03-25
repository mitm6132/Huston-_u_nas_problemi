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
        self.target_distance = 1.0
        self.mission_duration = 60.0

        # Управление
        self.kp_y = 0.003
        self.kp_dist = 0.35

        self.max_speed_y = 0.3
        self.max_speed_x = 0.3

        self.center_threshold = 50

        # Поиск
        self.search_speed = 0.2
        self.search_step_time = 2.0
        self.lost_wait = 1.5

        # Состояния
        self.state = 'SEARCH'
        self.selected_target = None
        self.lost_start = None
        self.search_direction = 1

        # AVOID
        self.avoid_direction = 1

        # Сглаживание
        self.cx_filtered = None

        # Таймеры
        self.last_search_switch = time.time()
        self.last_cmd_time = 0
        self.cmd_interval = 0.2

    # ------------------ CALLBACKS ------------------

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

    # ------------------ ОБНАРУЖЕНИЕ ПРЕПЯТСТВИЯ ------------------

    def detect_obstacle(self):
        if self.image is None:
            return False

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 60, 60])
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        h, w = mask.shape

        # анализируем центральную область (важно!)
        roi = mask[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

        ratio = cv2.countNonZero(roi) / roi.size

        return ratio > 0.05

    # ------------------ НАВИГАЦИЯ ------------------

    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5,
                      frame_id='aruco_map', tolerance=0.2, auto_arm=False):
        res = self.navigate(x=x, y=y, z=z, yaw=yaw, speed=speed,
                            frame_id=frame_id, auto_arm=auto_arm)

        if not res.success:
            return False

        while not rospy.is_shutdown():
            telem = self.get_telemetry(frame_id='navigate_target')
            dist = math.sqrt(telem.x**2 + telem.y**2 + telem.z**2)
            if dist < tolerance:
                return True
            rospy.sleep(0.1)

        return False

    def hold_height(self):
        telem = self.get_telemetry(frame_id='aruco_map')
        err = self.height - telem.z
        return max(-0.2, min(0.2, 0.8 * err))

    def send_cmd(self, x, y):
        z = self.hold_height()
        now = time.time()

        if now - self.last_cmd_time > self.cmd_interval:
            self.navigate(x=x, y=y, z=z,
                          yaw=float('nan'),
                          speed=0.5,
                          frame_id='body')
            self.last_cmd_time = now

    # ------------------ ЛОГИКА ------------------

    def select_target(self):
        if self.detections is None:
            return None

        persons = [d for d in self.detections.detections if d.class_name == "person"]

        if not persons:
            return None

        return max(persons, key=lambda d: (d.x_max - d.x_min)*(d.y_max - d.y_min))

    def get_center(self, t):
        return (t.x_min + t.x_max) // 2

    def iou(self, a, b):
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)

        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a.x_max-a.x_min)*(a.y_max-a.y_min)
        area_b = (b.x_max-b.x_min)*(b.y_max-b.y_min)

        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    # ------------------ MAIN ------------------

    def run(self):

        rospy.loginfo("Взлёт")
        if not self.navigate_wait(z=self.height, frame_id='body', auto_arm=True):
            return

        rospy.loginfo("Стабилизация")
        rospy.sleep(3)

        start = time.time()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            if time.time() - start > self.mission_duration:
                break

            obstacle = self.detect_obstacle()

            # ---------- AVOID ----------
            if obstacle:
                self.state = 'AVOID'

            if self.state == 'AVOID':
                rospy.loginfo_throttle(1, "AVOID")

                move_x = 0.0
                move_y = 0.3 * self.avoid_direction

                self.send_cmd(move_x, move_y)

                # если упёрлись — меняем сторону
                if self.range and self.range < 1.0:
                    self.avoid_direction *= -1

                if not obstacle:
                    self.state = 'TRACK'

            # ---------- SEARCH ----------
            elif self.state == 'SEARCH':

                target = self.select_target()

                if target:
                    self.selected_target = target
                    self.state = 'TRACK'
                    continue

                if time.time() - self.last_search_switch > self.search_step_time:
                    self.search_direction *= -1
                    self.last_search_switch = time.time()

                self.send_cmd(0, self.search_speed * self.search_direction)

            # ---------- TRACK ----------
            elif self.state == 'TRACK':

                target = self.select_target()

                if target is None:
                    if self.lost_start is None:
                        self.lost_start = time.time()

                    if time.time() - self.lost_start > self.lost_wait:
                        self.state = 'SEARCH'
                    else:
                        self.send_cmd(0, 0)
                    continue

                self.lost_start = None

                cx = self.get_center(target)

                # сглаживание
                if self.cx_filtered is None:
                    self.cx_filtered = cx
                else:
                    self.cx_filtered = 0.7*self.cx_filtered + 0.3*cx

                err = self.cx_filtered - self.image_width/2

                if abs(err) < 20:
                    err = 0

                move_y = -self.kp_y * err
                move_y = max(-self.max_speed_y, min(self.max_speed_y, move_y))

                move_x = 0

                if abs(err) < self.center_threshold and self.range:
                    dist_err = self.range - self.target_distance
                    move_x = self.kp_dist * dist_err

                    if self.range < 0.5:
                        move_x = 0

                self.send_cmd(move_x, move_y)

            rate.sleep()

        rospy.loginfo("Посадка")
        self.land()


if __name__ == '__main__':
    try:
        TargetTracker().run()
    except rospy.ROSInterruptException:
        pass