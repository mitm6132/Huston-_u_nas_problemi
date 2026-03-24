#!/usr/bin/env python3

import rospy
import math
import time
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range
from clover_yolo.msg import DetectionArray, Detection
from geometry_msgs.msg import Point

class TargetTracker:
    def __init__(self):
        rospy.init_node('target_tracker')
        
        # Сервисы Clover
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
        self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
        self.set_yaw = rospy.ServiceProxy('set_yaw', srv.SetYaw)
        self.land = rospy.ServiceProxy('land', Trigger)
        
        # Параметры
        self.target_distance = 1.5  # Заданная дистанция до цели (м)
        self.working_height = 1.5   # Рабочая высота (м)
        self.min_safe_distance = 0.5  # Минимальная безопасная дистанция (м)
        self.tracking_time = 20     # Время сопровождения (с)
        self.max_attempts = 300     # Максимальное время попытки (с)
        
        # Переменные состояния
        self.target_id = None
        self.current_target = None
        self.is_tracking = False
        self.tracking_start_time = None
        self.lost_counter = 0
        self.max_lost_frames = 10
        
        # Параметры управления
        self.kp_distance = 0.5      # Коэффициент регулятора дистанции
        self.kp_yaw = 1.5           # Коэффициент регулятора курса
        self.max_speed = 0.5        # Максимальная скорость (м/с)
        self.max_yaw_rate = 45      # Максимальная угловая скорость (град/с)
        
        # Подписки
        self.detections_sub = rospy.Subscriber('/vision/detections', DetectionArray, self.detections_callback)
        self.rangefinder_sub = rospy.Subscriber('/front_rangefinder/range', Range, self.rangefinder_callback)
        
        # Текущие данные
        self.current_range = None
        self.current_detections = []
        
        # Состояния автомата
        self.state = "INIT"  # INIT, TAKEOFF, DETECTING, TRACKING, LOST, RETURN, LAND
        
        # Время начала миссии
        self.start_time = time.time()
        
        rospy.loginfo("Target Tracker инициализирован")
    
    def detections_callback(self, msg):
        """Обработка детекций от нейросети"""
        self.current_detections = msg.detections
        
        # Для отладки
        if len(msg.detections) > 0:
            rospy.logdebug(f"Получено {len(msg.detections)} детекций")
            for det in msg.detections:
                rospy.logdebug(f"ID: {det.id}, Class: {det.class_name}, Score: {det.score:.2f}")
    
    def rangefinder_callback(self, msg):
        """Обработка данных дальномера"""
        self.current_range = msg.range
        rospy.logdebug(f"Текущая дистанция: {self.current_range:.2f} м")
    
    def select_target(self):
        """Выбор целевой фигуры человека"""
        target_persons = [d for d in self.current_detections if d.class_name == 'target_person']
        
        if not target_persons:
            rospy.logwarn("Целевые фигуры не обнаружены")
            return None
        
        # Выбираем фигуру с максимальным score в центре кадра
        best_target = None
        best_score = 0
        
        for person in target_persons:
            # Приоритет: чем ближе к центру и выше score
            center_x = person.bbox.center.x
            center_y = person.bbox.center.y
            center_score = 1.0 - math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            total_score = person.score * (0.7 + 0.3 * center_score)
            
            if total_score > best_score:
                best_score = total_score
                best_target = person
        
        if best_target:
            rospy.loginfo(f"Выбрана цель ID: {best_target.id}, Score: {best_target.score:.2f}, Class: {best_target.class_name}")
            return best_target
        return None
    
    def calculate_control(self, detection, current_range):
        """Расчет управляющих воздействий на основе детекции и дальномера"""
        if detection is None or current_range is None:
            return 0, 0, 0  # velocity_x, velocity_y, yaw_rate
        
        # Ошибка по дистанции
        distance_error = current_range - self.target_distance
        velocity_forward = -self.kp_distance * distance_error
        velocity_forward = max(min(velocity_forward, self.max_speed), -self.max_speed)
        
        # Ошибка по положению в кадре (центрирование)
        center_x = detection.bbox.center.x
        center_error = center_x - 0.5
        yaw_rate = self.kp_yaw * center_error * self.max_yaw_rate
        yaw_rate = max(min(yaw_rate, self.max_yaw_rate), -self.max_yaw_rate)
        
        # Горизонтальное выравнивание
        center_y = detection.bbox.center.y
        height_error = center_y - 0.5
        velocity_vertical = 0  # Высота фиксирована
        
        return velocity_forward, 0, yaw_rate  # vx, vy, yaw_rate
    
    def check_target_in_frame(self, detection):
        """Проверка, находится ли цель в центре кадра"""
        if detection is None:
            return False
        
        center_x = detection.bbox.center.x
        center_y = detection.bbox.center.y
        
        # Цель считается в кадре, если находится в центральной области
        return abs(center_x - 0.5) < 0.3 and abs(center_y - 0.5) < 0.3
    
    def run(self):
        """Основной цикл управления"""
        rate = rospy.Rate(20)  # 20 Гц
        
        while not rospy.is_shutdown():
            current_time = time.time()
            
            # Проверка времени миссии
            if current_time - self.start_time > self.max_attempts:
                rospy.logerr("Превышено максимальное время миссии")
                self.state = "LAND"
            
            if self.state == "INIT":
                rospy.loginfo("Начало миссии")
                self.state = "TAKEOFF"
            
            elif self.state == "TAKEOFF":
                rospy.loginfo("Выполняется взлёт...")
                self.navigate(x=0, y=0, z=self.working_height, speed=0.5, 
                            frame_id='body', auto_arm=True)
                rospy.sleep(5)
                self.state = "DETECTING"
            
            elif self.state == "DETECTING":
                if self.current_detections:
                    target = self.select_target()
                    if target:
                        self.target_id = target.id
                        self.current_target = target
                        self.state = "TRACKING"
                        self.tracking_start_time = current_time
                        rospy.loginfo(f"Цель обнаружена, начинаем сопровождение (ID: {self.target_id})")
                    else:
                        rospy.loginfo("Поиск цели...")
                        # Медленное вращение для поиска
                        self.set_yaw(yaw_rate=15, frame_id='body')
                else:
                    rospy.loginfo("Ожидание детекций...")
                rospy.sleep(0.1)
            
            elif self.state == "TRACKING":
                # Проверка времени сопровождения
                if current_time - self.tracking_start_time >= self.tracking_time:
                    rospy.loginfo("Сопровождение завершено успешно")
                    self.state = "RETURN"
                    continue
                
                # Поиск текущей цели среди детекций
                current_target = None
                for det in self.current_detections:
                    if det.id == self.target_id and det.class_name == 'target_person':
                        current_target = det
                        break
                
                if current_target:
                    self.lost_counter = 0
                    self.current_target = current_target
                    
                    # Проверка дальномера
                    if self.current_range is not None:
                        # Проверка безопасной дистанции
                        if self.current_range < self.min_safe_distance:
                            rospy.logwarn("Слишком близко к цели, отлетаем назад")
                            self.set_velocity(vx=-0.3, vy=0, vz=0, yaw_rate=0, frame_id='body')
                        else:
                            # Расчет управления
                            vx, vy, yaw_rate = self.calculate_control(current_target, self.current_range)
                            
                            # Вывод отладочной информации
                            rospy.loginfo(f"Tracking: Dist={self.current_range:.2f}m, "
                                        f"Yaw_rate={yaw_rate:.1f}deg/s, "
                                        f"Score={current_target.score:.2f}")
                            
                            # Отправка команды
                            self.set_velocity(vx=vx, vy=vy, vz=0, yaw_rate=yaw_rate, 
                                            frame_id='body')
                    else:
                        rospy.logwarn("Нет данных дальномера")
                        # Только поворот на цель
                        if current_target:
                            center_error = current_target.bbox.center.x - 0.5
                            yaw_rate = self.kp_yaw * center_error * self.max_yaw_rate
                            self.set_velocity(vx=0, vy=0, vz=0, yaw_rate=yaw_rate, 
                                            frame_id='body')
                else:
                    self.lost_counter += 1
                    rospy.logwarn(f"Цель потеряна, кадров без детекции: {self.lost_counter}")
                    
                    if self.lost_counter >= self.max_lost_frames:
                        self.state = "LOST"
                    else:
                        # Попытка восстановления - медленное вращение
                        self.set_velocity(vx=0, vy=0, vz=0, yaw_rate=10, frame_id='body')
            
            elif self.state == "LOST":
                rospy.loginfo("Потеря цели, пытаемся восстановить...")
                
                # Поиск цели среди всех детекций
                target = self.select_target()
                if target:
                    self.target_id = target.id
                    self.current_target = target
                    self.lost_counter = 0
                    self.state = "TRACKING"
                    rospy.loginfo(f"Цель восстановлена (ID: {self.target_id})")
                else:
                    # Медленное вращение для поиска
                    self.set_yaw(yaw_rate=15, frame_id='body')
                    rospy.sleep(0.1)
            
            elif self.state == "RETURN":
                rospy.loginfo("Возврат на базу...")
                # Возврат в стартовую позицию
                self.navigate(x=0, y=0, z=self.working_height, speed=0.5, 
                            frame_id='map', auto_arm=False)
                rospy.sleep(5)
                self.state = "LAND"
            
            elif self.state == "LAND":
                rospy.loginfo("Выполняется посадка...")
                self.land()
                rospy.signal_shutdown("Миссия завершена")
                break
            
            rate.sleep()

if __name__ == '__main__':
    try:
        tracker = TargetTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass