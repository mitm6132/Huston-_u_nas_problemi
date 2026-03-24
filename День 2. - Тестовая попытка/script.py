#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range, CameraInfo
from clover_yolo.msg import DetectionArray, Detection

class TargetTracker:
    def __init__(self):
        rospy.init_node('target_tracker', anonymous=True)

        # Сервисы Clover
        rospy.wait_for_service('navigate')
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('land')
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.land = rospy.ServiceProxy('land', Trigger)

        # Подписки
        self.detections = None
        self.range = None
        self.image_width = 640
        self.image_height = 480
        rospy.Subscriber('/front_vision/detections', DetectionArray, self.detections_cb)
        rospy.Subscriber('/front_rangefinder/range', Range, self.range_cb)
        rospy.Subscriber('/front_main_camera/camera_info', CameraInfo, self.camera_info_cb)

        # Параметры миссии
        self.height = 1.5                 # высота, м
        self.target_distance = 1.5        # желаемая дистанция до цели, м
        self.mission_duration = 120.0     # общая длительность миссии, сек

        # Управление движением
        self.kp_y = 0.005                 # коэффициент для центрирования (Y)
        self.kp_dist = 0.8                # коэффициент для дистанции (X)
        self.max_speed_y = 0.3            # макс. скорость влево/вправо, м/с
        self.max_speed_x = 0.3            # макс. скорость вперёд/назад, м/с
        self.center_threshold = 50        # пикселей – считаем цель в центре

        # Параметры поиска
        self.search_speed = 0.2           # скорость движения при поиске, м/с
        self.search_step_time = 2.0       # время движения в одном направлении, сек
        self.lost_wait = 1.5              # пауза перед началом поиска, сек

        # Состояние
        self.state = 'SEARCH'             # SEARCH, TRACK
        self.selected_target = None
        self.lost_start = None
        self.search_direction = 1

        # Ограничение частоты команд
        self.last_cmd_time = 0
        self.cmd_interval = 0.2

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def camera_info_cb(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5,
                      frame_id='aruco_map', tolerance=0.2, auto_arm=False):
        res = self.navigate(x=x, y=y, z=z, yaw=yaw, speed=speed,
                            frame_id=frame_id, auto_arm=auto_arm)
        if not res.success:
            rospy.logerr("navigate failed: " + res.message)
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
        err_z = self.height - telem.z
        move_z = 0.8 * err_z
        move_z = max(-0.2, min(0.2, move_z))
        return move_z

    def select_target(self):
        if self.detections is None or not self.detections.detections:
            return None
        persons = [d for d in self.detections.detections if d.class_name == "person"]
        if not persons:
            return None
        return max(persons, key=lambda d: (d.x_max - d.x_min)*(d.y_max - d.y_min))

    def get_center(self, target):
        return (target.x_min + target.x_max) // 2, (target.y_min + target.y_max) // 2

    def iou(self, a, b):
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x_max - a.x_min)*(a.y_max - a.y_min)
        area_b = (b.x_max - b.x_min)*(b.y_max - b.y_min)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def send_cmd(self, move_x, move_y):
        move_z = self.hold_height()
        now = time.time()
        if now - self.last_cmd_time > self.cmd_interval:
            self.navigate(x=move_x, y=move_y, z=move_z, yaw=float('nan'),
                          speed=0.5, frame_id='body', auto_arm=False)
            self.last_cmd_time = now

    def run(self):
        # --- Взлёт ---
        rospy.loginfo("Взлёт на %.1f м", self.height)
        if not self.navigate_wait(z=self.height, frame_id='body', auto_arm=True):
            return

        # --- Перелёт в точку (2,2) по aruco_map ---
        rospy.loginfo("Перелёт в точку (2,2)")
        if not self.navigate_wait(x=2, y=2, z=self.height, frame_id='aruco_map', speed=0.5):
            rospy.logwarn("Не удалось достичь (2,2), продолжаем...")

        # --- Стабилизация ---
        rospy.loginfo("Стабилизация 10 секунд...")
        start_stable = time.time()
        while time.time() - start_stable < 10.0:
            self.send_cmd(0.0, 0.0)
            rospy.sleep(0.2)

        # --- Основной цикл миссии ---
        mission_start = time.time()
        rospy.loginfo("Миссия началась. Длительность: %.0f секунд", self.mission_duration)
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            elapsed = time.time() - mission_start
            if elapsed >= self.mission_duration:
                rospy.loginfo("Время миссии истекло (%.0f с)", elapsed)
                break

            if self.state == 'SEARCH':
                target = self.select_target()
                if target is not None:
                    rospy.loginfo("Цель обнаружена, переходим в TRACK")
                    self.selected_target = target
                    self.state = 'TRACK'
                    self.lost_start = None
                    continue

                # Поиск: движение влево-вправо
                move_y = self.search_speed * self.search_direction
                self.send_cmd(0.0, move_y)
                rospy.sleep(self.search_step_time)
                self.search_direction *= -1
                rospy.loginfo_throttle(5, "Поиск: смена направления")

            elif self.state == 'TRACK':
                if self.detections is None:
                    self.send_cmd(0.0, 0.0)
                    rate.sleep()
                    continue

                # Найти текущую цель по IOU
                current_target = None
                for d in self.detections.detections:
                    if d.class_name != "person":
                        continue
                    if self.selected_target is not None:
                        if self.iou(self.selected_target, d) > 0.5:
                            current_target = d
                            break
                    else:
                        current_target = d
                        break

                if current_target is None:
                    # Потеря цели
                    if self.lost_start is None:
                        self.lost_start = time.time()
                        rospy.loginfo("Цель потеряна, ждём %.1f с...", self.lost_wait)
                    if time.time() - self.lost_start > self.lost_wait:
                        rospy.loginfo("Переход в режим поиска")
                        self.state = 'SEARCH'
                        self.lost_start = None
                        self.search_direction = 1
                    else:
                        self.send_cmd(0.0, 0.0)
                else:
                    self.lost_start = None
                    if self.selected_target is None:
                        self.selected_target = current_target
                        rospy.loginfo("Слежение возобновлено")

                    # Управление центрированием (Y)
                    cx, _ = self.get_center(current_target)
                    err_x = cx - self.image_width / 2
                    move_y = - self.kp_y * err_x
                    move_y = max(-self.max_speed_y, min(self.max_speed_y, move_y))

                    # Управление дистанцией (X) – только если цель в центре
                    move_x = 0.0
                    if abs(err_x) < self.center_threshold and self.range is not None:
                        err_dist = self.range - self.target_distance
                        move_x = self.kp_dist * err_dist
                        move_x = max(-self.max_speed_x, min(self.max_speed_x, move_x))

                    self.send_cmd(move_x, move_y)

                    rospy.loginfo_throttle(1,
                        "Слежение: дист=%.2f м, errX=%.0f пикс, move_x=%.2f, move_y=%.2f, время=%.1f с",
                        self.range if self.range else 0, err_x, move_x, move_y, elapsed)

            rate.sleep()

        # --- Возврат на старт и посадка ---
        rospy.loginfo("Возврат на старт")
        self.navigate_wait(x=0, y=0, z=self.height, frame_id='aruco_map', speed=0.5)
        rospy.loginfo("Посадка")
        self.land()
        rospy.loginfo("Миссия завершена")

if __name__ == '__main__':
    try:
        tracker = TargetTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass