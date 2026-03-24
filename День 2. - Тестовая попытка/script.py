#!/usr/bin/env python3
import rospy
import math
import time
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range, CameraInfo
from clover_yolo.msg import DetectionArray, Detection

class PersonTracker:
    def __init__(self):
        rospy.init_node('person_tracker', anonymous=True)

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
        rospy.Subscriber('/vision/detections', DetectionArray, self.detections_cb)
        rospy.Subscriber('/front_rangefinder/range', Range, self.range_cb)
        rospy.Subscriber('/front_main_camera/camera_info', CameraInfo, self.camera_info_cb)

        # Параметры
        self.target_distance = 0.5      # целевая дистанция (м)
        self.tracking_time = 120.0       # время удержания (с)
        self.height = 1.5               # фиксированная высота (м)
        self.speed_fwd = 0.3            # максимальная скорость вперёд/назад

        # Коэффициент регулятора дистанции
        self.kp_dist = 0.8

        # Состояние
        self.selected_target = None
        self.tracking_start = None
        self.last_height_keep_time = 0

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def camera_info_cb(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5,
                      frame_id='body', tolerance=0.2, auto_arm=False):
        """Ожидание завершения движения (для взлёта и возврата)"""
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

    def select_target(self):
        """Выбрать человека с самой большой площадью bounding box"""
        if self.detections is None or not self.detections.detections:
            return None
        persons = [d for d in self.detections.detections if d.class_name == "person"]
        if not persons:
            return None
        return max(persons, key=lambda d: (d.x_max - d.x_min)*(d.y_max - d.y_min))

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

    def keep_height(self):
        """Поддержание высоты (раз в секунду)"""
        now = time.time()
        if now - self.last_height_keep_time >= 1.0:
            self.navigate(x=0, y=0, z=self.height, speed=0.5, frame_id='map', auto_arm=False)
            self.last_height_keep_time = now

    def run(self):
        # Взлёт
        rospy.loginfo("Взлёт на %.1f м", self.height)
        if not self.navigate_wait(z=self.height, frame_id='body', auto_arm=True):
            return
        
        self.navigate_wait(x = 2, y = 2, frame_id="aruco_map")

        # Ждём появления человека
        rospy.loginfo("Ожидание человека...")
        while not rospy.is_shutdown():
            target = self.select_target()
            if target is not None:
                self.selected_target = target
                rospy.loginfo("Цель выбрана")
                break
            rospy.sleep(0.2)

        # Подлёт на целевую дистанцию
        rospy.loginfo("Подлёт на %.1f м", self.target_distance)
        while not rospy.is_shutdown():
            if self.range is None:
                rospy.sleep(0.1)
                continue
            err = self.range - self.target_distance
            if abs(err) < 0.2:
                break
            move = self.kp_dist * err
            move = max(-self.speed_fwd, min(self.speed_fwd, move))
            self.navigate(x=move, y=0, z=0, speed=0.5, frame_id='body', auto_arm=False)
            self.keep_height()
            rospy.sleep(0.1)

        # Основной цикл сопровождения
        rospy.loginfo("Начинаем сопровождение")
        self.tracking_start = time.time()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            # Ищем текущую цель по IOU (если есть)
            current_target = None
            if self.detections:
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

            # Если цель потеряна, пытаемся найти любого человека
            if current_target is None:
                candidate = self.select_target()
                if candidate is not None:
                    rospy.loginfo("Цель восстановлена")
                    self.selected_target = candidate
                    current_target = candidate
                else:
                    # Нет детекций — ждём, не двигаясь
                    self.keep_height()
                    rate.sleep()
                    continue
            else:
                if self.selected_target is None:
                    self.selected_target = current_target
                    self.tracking_start = time.time()
                    rospy.loginfo("Сопровождение начато")

            # Управление по дальности
            if self.range is not None:
                err_dist = self.range - self.target_distance
                move_x = self.kp_dist * err_dist
                move_x = max(-self.speed_fwd, min(self.speed_fwd, move_x))
            else:
                move_x = 0

            # Отправляем команду движения вперёд/назад
            self.navigate(x=move_x, y=0, z=0, speed=0.5, frame_id='body', auto_arm=False)

            # Поддерживаем высоту
            self.keep_height()

            # Логирование (раз в секунду)
            rospy.loginfo_throttle(1,
                "Дист: %.2f м | MoveX: %.2f м/с",
                self.range if self.range else 0, move_x)

            # Проверка времени сопровождения
            if time.time() - self.tracking_start >= self.tracking_time:
                rospy.loginfo("Сопровождение завершено (%.1f с)", self.tracking_time)
                break

            rate.sleep()

        # Возврат на старт и посадка
        rospy.loginfo("Возврат на старт")
        self.navigate_wait(x=0, y=0, z=self.height, frame_id='aruco_map', speed=0.5)
        rospy.loginfo("Посадка")
        self.land()
        rospy.loginfo("Миссия завершена")

if __name__ == '__main__':
    try:
        PersonTracker().run()
    except rospy.ROSInterruptException:
        pass