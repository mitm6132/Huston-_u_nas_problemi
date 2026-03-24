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
        rospy.wait_for_service('set_yaw')
        rospy.wait_for_service('land')
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.set_yaw = rospy.ServiceProxy('set_yaw', srv.SetYaw)
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
        self.target_distance = 1.5      # целевая дистанция (м)
        self.tracking_time = 120.0      # время удержания (с)
        self.height = 1.5               # высота полёта (м)
        self.speed_fwd = 0.3            # скорость вперёд/назад
        
        # Коэффициент регулятора дистанции
        self.kp_dist = 0.8
        
        # Состояние
        self.selected_target = None
        self.tracking_start = None
        self.last_height_keep_time = 0
        self.current_yaw = 0

    def detections_cb(self, msg):
        self.detections = msg

    def range_cb(self, msg):
        self.range = msg.range

    def camera_info_cb(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5,
                      frame_id='body', tolerance=0.2, auto_arm=False):
        """Ожидание завершения движения"""
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
        """Intersection over Union для отслеживания цели"""
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

    def search_for_person(self):
        """Поиск человека с поворотами на 45 градусов"""
        rospy.loginfo("Ищу человека...")
        
        # Делаем 8 поворотов по 45 градусов (полный круг)
        for i in range(8):
            self.current_yaw += 45
            rospy.loginfo(f"Поворот {i+1}/8: на {self.current_yaw}°")
            
            # Поворачиваемся с ожиданием
            self.navigate_wait(yaw=self.current_yaw, frame_id='map', tolerance=5)
            rospy.sleep(1)
            
            # Проверяем, видим ли человека
            target = self.select_target()
            if target is not None:
                rospy.loginfo(f"Нашёл человека! Поворот {i+1}")
                return target
        
        rospy.logwarn("Человек не найден после полного оборота")
        return None

    def fly_to_point(self, x, y):
        """Летим к координатам с использованием navigate_wait"""
        rospy.loginfo(f"Летим к точке: x={x}, y={y}")
        
        # Летим по координатам
        success = self.navigate_wait(x=x, y=y, z=self.height, speed=0.5, 
                                     frame_id='aruco_map', auto_arm=False)
        
        if success:
            rospy.loginfo(f"Точка x={x}, y={y} достигнута")
        else:
            rospy.logerr(f"Не удалось долететь до точки x={x}, y={y}")
        
        return success

    def run(self):
        # Взлёт
        rospy.loginfo(f"Взлёт на {self.height} м")
        if not self.navigate_wait(z=self.height, frame_id='body', auto_arm=True):
            return
        
        rospy.sleep(2)
        
        # Летим к координатам x=2, y=2
        rospy.loginfo("Летим к координатам x=2, y=2")
        if not self.fly_to_point(x=2, y=2):
            rospy.logerr("Не удалось долететь до целевой точки, сажусь...")
            self.land()
            return
        
        # Поиск человека с поворотами
        target = self.search_for_person()
        if target is None:
            rospy.logerr("Человек не найден, сажусь...")
            self.land()
            return
        
        self.selected_target = target
        rospy.loginfo("Цель выбрана, начинаю сопровождение на 120 секунд")
        
        # Основной цикл сопровождения
        self.tracking_start = time.time()
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            # Проверка времени
            elapsed = time.time() - self.tracking_start
            if elapsed >= self.tracking_time:
                rospy.loginfo(f"Сопровождение завершено ({self.tracking_time} сек)")
                break
            
            # Ищем текущую цель по IOU
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
            
            # Если цель потеряна, ищем любого человека
            if current_target is None:
                new_target = self.select_target()
                if new_target is not None:
                    rospy.loginfo("Цель восстановлена")
                    self.selected_target = new_target
                    current_target = new_target
                else:
                    # Медленно вращаемся в поиске
                    self.current_yaw += 10
                    self.navigate_wait(yaw=self.current_yaw, frame_id='aruco_map', tolerance=5)
                    self.keep_height()
                    rate.sleep()
                    continue
            else:
                self.selected_target = current_target
            
            # Управление по дальности (движение вперёд/назад)
            if self.range is not None:
                err_dist = self.range - self.target_distance
                move_x = self.kp_dist * err_dist
                move_x = max(-self.speed_fwd, min(self.speed_fwd, move_x))
                
                # Двигаемся вперёд/назад относительно текущей позиции
                if abs(move_x) > 0.05:  # Двигаемся только если ошибка значительная
                    self.navigate(x=move_x, y=0, z=0, speed=0.3, 
                                frame_id='body', auto_arm=False)
                
                # Логирование каждые 5 секунд
                if int(elapsed) % 5 == 0:
                    rospy.loginfo(
                        f"Сопровождение: {elapsed:.0f}/{self.tracking_time} сек | "
                        f"Дист: {self.range:.2f}м | MoveX: {move_x:.2f}")
            else:
                rospy.logwarn_throttle(1, "Нет данных дальномера")
            
            # Поддерживаем высоту
            self.keep_height()
            rate.sleep()
        
        # Возврат на старт (0,0) и посадка
        rospy.loginfo("Возврат на старт (0,0)")
        self.navigate_wait(x = 0, y = 0, frame_id='aruco_map')

        rospy.loginfo("Посадка")
        self.land()
        rospy.loginfo("Миссия завершена")

if __name__ == '__main__':
    try:
        PersonTracker().run()
    except rospy.ROSInterruptException:
        pass