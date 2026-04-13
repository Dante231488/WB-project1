from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import numpy as np
from ultralytics import YOLO
import cv2
import numpy as np
from torch.nn import PairwiseDistance
from collections import deque
from datetime import datetime
import os
import torch
from scipy import interpolate
import numpy as np
from scipy.interpolate import BSpline

from scipy.spatial import distance


class Config:
    # HSV диапазоны (будут меняться ползунками)
    HUE_RANGES = {
        'blue': {'min': 69, 'max': 130, 'bgr': (255, 0, 0), 'name': 'СИНИЙ (левая рука)'},
        'green': {'min': 30, 'max': 67, 'bgr': (0, 255, 0), 'name': 'ЗЕЛЁНЫЙ (правая рука)'},
        'yellow': {'min': 17, 'max': 31, 'bgr': (0, 255, 255), 'name': 'ЖЁЛТЫЙ (туловище+ноги)'}
    }

    S_MIN, S_MAX = 100, 255
    V_MIN, V_MAX = 100, 255

    # Параметры детекции
    MIN_MARKER_AREA = 300
    MAX_AREA  =8000
    HISTORY_SECONDS = 4
    FPS = 30
    PREDICTION_HORIZON = 15  # предсказание на 0.5 секунды

    @classmethod
    def get_history_size(cls):
        return cls.HISTORY_SECONDS * cls.FPS



class MarkerTrack:
    """Отслеживает один маркер с историей движения"""

    def __init__(self, track_id, color_name, x, y):
        self.id = track_id
        self.color_name = color_name
        self.color_bgr = Config.HUE_RANGES[color_name]['bgr']
        self.positions = deque(maxlen=Config.get_history_size())
        self.positions.append((x, y))
        self.age = 0

    def update(self, x, y):
        self.positions.append((x, y))
        self.age = 0

    def increment_age(self):
        self.age += 1

    def is_alive(self):
        return self.age <= 10

    def get_trajectory(self):
        return list(self.positions)

    def get_smoothed_trajectory(self, window=4):
        """Сглаженная траектория для визуализации"""
        traj = list(self.positions)
        if len(traj) < window:
            return list(self.positions)
        smoothed = []
       
        

        for i in range(len(traj)):
            start = max(0, i - window // 2)
            end = min(len(traj), i + window // 2 + 1)
            window_pts = traj[start:end]
            avg_x = int(np.mean([p[0] for p in window_pts]))
            avg_y = int(np.mean([p[1] for p in window_pts]))
            smoothed.append((avg_x, avg_y))
        return smoothed
        




        
        




class TrajectoryPredictor:


    @staticmethod
    def compute_rmse(predicted, actual):
        """Вычисляет RMSE для оценки качества"""
        if not predicted or not actual:
            return float('inf')
        min_len = min(len(predicted), len(actual))
        pred = np.array(predicted[:min_len])
        act = np.array(actual[:min_len])
        return np.sqrt(np.mean((pred - act) ** 2))




def get_centers(mask, min_area=Config.MIN_MARKER_AREA,max_area = Config.MAX_AREA):
    """Находит центры маркеров с фильтрацией по площади"""
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area<max_area :
            # Проверка округлости
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:  # Слишком вытянутый
                    continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers


def detect_all(hsv, trackbars):
    """Детектирует маркеры всех цветов"""
    detected = {}
    masks = {}

    s_min = cv2.getTrackbarPos('S Min', 'Hue Settings')
    s_max = cv2.getTrackbarPos('S Max', 'Hue Settings')
    v_min = cv2.getTrackbarPos('V Min', 'Hue Settings')
    v_max = cv2.getTrackbarPos('V Max', 'Hue Settings')

    for color_name in Config.HUE_RANGES:
        h_min = cv2.getTrackbarPos(f'{color_name} H Min', 'Hue Settings')
        h_max = cv2.getTrackbarPos(f'{color_name} H Max', 'Hue Settings')

        if h_min > h_max:
            h_min, h_max = h_max, h_min

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)

        # Морфологическая очистка
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        masks[color_name] = mask
        points = get_centers(mask)
        detected[color_name] = sorted(points, key=lambda p: p[1])  # сортируем сверху вниз

    return detected, masks



class Tracker:
    def __init__(self):
        self.tracks = {}  # {(color_name, idx): MarkerTrack}
        self.next_ids = {color: 0 for color in Config.HUE_RANGES.keys()}

    def update(self, detections):
        """Обновляет треки на основе новых детекций"""
        new_tracks = {}
        det = []
        old = []
        t_id1 = []
        color_new= []
        color_old = []
        for color_name, centers in detections.items():
            for (x, y) in centers:
                color_new.append(color_name)
                det.append([x,y])


        for (c_name, t_id), track in self.tracks.items():
                    if not track.positions:
                        continue
                    last_x, last_y = track.positions[-1]
                    t_id1.append(t_id)
                    color_old.append(c_name)
                    old.append([last_x,last_y])

        det  =np.array(det,dtype=np.float32)
        old = np.array(old,dtype=np.float32)
        

        used = [0]*len(det)
        matches = {}
        if len(det)>0 and len(old)>0:
          
          #cost_mtr = distance.cdist(old,det, metric='euclidean') # cost_mtr.shape=(n,m)
          cost_mtr = (torch.cdist(torch.tensor(old,device=device,dtype=torch.float32),torch.tensor(det,device=device,dtype=torch.float32),p = 2)).cpu().detach().numpy()
          print(cost_mtr)
          i,j = linear_sum_assignment(cost_mtr)
          for i, j in list(zip(i, j)):
            matches[j] = i
            used[j]=1
        
        for j in range(0,len(det)):
          if used[j]==1:
              
              track = self.tracks[(color_old[int(matches[j])], t_id1[int(matches[j])])]
              track.update(int(det[j][0]),int(det[j][1]))
              
              new_tracks[(color_old[int(matches[j])], t_id1[int(matches[j])])] = track
          else:
            new_id = self.next_ids[color_new[j]]
            self.next_ids[color_new[j]] += 1
            new_tracks[(color_new[j], new_id)] = MarkerTrack(new_id, color_new[j], int(det[j][0]), int(det[j][1]))
        
        for key, track in self.tracks.items():
            if key not in new_tracks:
                track.increment_age()
                if track.is_alive():
                    pred = track.predict() if hasattr(track, 'predict') else track.positions[-1]
                    track.positions.append(pred if isinstance(pred, tuple) else track.positions[-1])
                    new_tracks[key] = track

        self.tracks = new_tracks
        return self.tracks


def draw_trajectories(frame, tracks):
    """Рисует траектории движения маркеров"""
    for (color_name, track_id), track in tracks.items():
        color = track.color_bgr
        traj = track.get_smoothed_trajectory()
        t = 20
        # Рисуем историю с градиентом
        if len(traj) > t:
            for i in range(len(traj)-t, len(traj)):
                alpha = i / len(traj)
                trail_color = (int(255 * alpha), int(255 * (1 - alpha)), 0)
                cv2.line(frame, traj[i - 1], traj[i], trail_color, 2)
    return frame
def predict(d,t):
       res2 = d[len(d)-t:len(d)]
       res2 = np.array(res2,dtype=float)
       
       
       last = res2[-1,0]
       pr = res2[-2,0]
       res2 = np.array(sorted(res2.tolist(),key=lambda p: p[0]))
       
       res2x= res2[:,0:1].reshape(t)
       res2y = res2[:,1:2].reshape(t)
       
       
       low = 0
       hi = 0
       if last>pr:
         low = res2x[-1]
         hi  = res2x[-1] + 10*abs(res2x[-1] - res2x[-2])
       else:
         low = res2x[0] - 10*abs(res2x[-1] - res2x[-2])
         hi  = res2x[-1]
         
       
       #res2x.sort()
       f = np.linspace(0.001,0.01,t)
       res2x+=f
       
       p = 10
       y = np.zeros((p,2))
       
       
       y[:,0] = np.linspace(low,hi,p)
       
       f = interpolate.interp1d(res2x, res2y, fill_value='extrapolate',kind='slinear')
       y[:,1] = f(y[:,0])
       
       return y.astype(int)


def draw_predictions(frame, tracks):
    """Рисует предсказанные траектории"""
    predictor = TrajectoryPredictor()
    t = 10
    for (color_name, track_id), track in tracks.items():
        traj = track.get_smoothed_trajectory()
        

        if len(traj) >= t:
            d = traj[len(traj)-t:len(traj)]
            predictions = predict(d,t)
            
            if len(predictions) > 1:
                for i in range(1, len(predictions)):
                    alpha = i / len(predictions)
                    color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
                    if(abs(predictions[i - 1][1]-predictions[i][1])<60):
                     cv2.line(frame, (predictions[i - 1][0],predictions[i - 1][1]), (predictions[i][0],predictions[i][1]), color, 2, cv2.LINE_AA)

                # Последняя предсказанная точка
                if len(predictions)>0:
                    cv2.circle(frame, predictions[-1], 6, (0, 0, 255), 3)

    return frame


def draw_current_markers(frame, detected):
    """Рисует текущие обнаруженные маркеры"""
    for color_name, centers in detected.items():
        color_bgr = Config.HUE_RANGES[color_name]['bgr']
        for (x, y) in centers:
            cv2.circle(frame, (x, y), 10, color_bgr, 2)
            cv2.circle(frame, (x, y), 4, color_bgr, -1)
    return frame

def draw1(frame,a):
    if(len(a)==0):
        return frame
    a = sorted(a,key=lambda p: p[1])
    for i in range(1,len(a)):
        cv2.circle(frame, a[i], 8, (35,220,240), 3)
        cv2.line(frame,a[i-1],a[i],(35,220,240),3)
    cv2.circle(frame,a[0],8,(35,220,240),3)
    return frame

def draw_skeleton(frame, detected):
    """Рисует скелет по анатомии"""
    blue = detected['blue']  # левая рука
    green = detected['green']  # правая рука
    yellow = sorted(detected['yellow'],key=lambda p: p[1])  # голова, шея, тело, ноги
    draw = yellow 
    neck = []
    legs = []
    leg1 = []
    leg2 = []
    
    for i in range(0,min(2,len(draw))):
        neck.append(draw[i])
    
    for i in range(2,min(8,len(draw))):
        legs.append(draw[i])
    if len(legs)>0:
        legs = sorted(legs,key = lambda p:p[0])
        for i in range(0,min(3,len(legs))):
            leg1.append(legs[i])
        for i in range(3,min(6,len(legs))):
            leg2.append(legs[i])
    if len(leg1)>0:
        if len(neck)>=2:
          leg1 = sorted(leg1,key = lambda p : p[1])
          cv2.line(frame,leg1[0],neck[-1],(255,255,255),3)
    
    if len(leg2)>0:
        if len(neck)>=2:
          leg2 = sorted(leg2,key = lambda p : p[1])
          cv2.line(frame, leg2[0],neck[-1],(255,255,255),3)
    
    frame = draw1(frame,neck)
    frame = draw1(frame,leg1)
    frame = draw1(frame,leg2)
    if len(blue)>0:
       blue = sorted(blue,key=lambda p: p[1])
       if len(neck)>=2:
           blue = sorted(blue,key=lambda p: p[0])
           cv2.line(frame,neck[1],blue[-1],(255,255,255),3)
       blue = sorted(blue,key=lambda p: p[1])
       for i in range(1,len(blue)):
          cv2.circle(frame, blue[i], 8, (255,0,0), 3)
          cv2.line(frame,blue[i-1],blue[i],(255,0,0),3)
       cv2.circle(frame,blue[0],8,(255,0,0),3)
    
    
        

    if len(green)>0:
       
       
       if len(neck)>=2:
           green = sorted(green,key=lambda p: p[0])
           cv2.line(frame,neck[1],green[0],(255,255,255),3)
       green = sorted(green,key=lambda p: p[1])
       for i in range(1,len(green)):
          cv2.circle(frame, green[i], 8, (0, 255, 0), 3)
          cv2.line(frame,green[i-1],green[i],(0,255,0),3)
       cv2.circle(frame,green[0],8,(0,255,0),3)
    return frame
 
   

  






def init_trackbars():
    cv2.namedWindow('Hue Settings')

    for color in Config.HUE_RANGES:
        cv2.createTrackbar(f'{color} H Min', 'Hue Settings', Config.HUE_RANGES[color]['min'], 180, lambda x: None)
        cv2.createTrackbar(f'{color} H Max', 'Hue Settings', Config.HUE_RANGES[color]['max'], 180, lambda x: None)

    cv2.createTrackbar('S Min', 'Hue Settings', Config.S_MIN, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'Hue Settings', Config.S_MAX, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'Hue Settings', Config.V_MIN, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'Hue Settings', Config.V_MAX, 255, lambda x: None)

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MARKER MOTION CAPTURE LITE - ПОЛНАЯ ВЕРСИЯ                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎨 ЦВЕТА МАРКЕРОВ:                                                          ║
║     🔵 СИНИЙ  - левая рука (3 маркера: плечо, локоть, кисть)                ║
║     🟢 ЗЕЛЁНЫЙ - правая рука (3 маркера: плечо, локоть, кисть)              ║
║     🟡 ЖЁЛТЫЙ  - голова, шея, бёдра, колени, стопы (всего 8)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✅ РЕАЛИЗОВАННЫЕ ТРЕБОВАНИЯ:                                                ║
║     1. Детекция маркеров 3 цветов                                           ║
║     2. Трекинг с историей 4 секунды                                         ║
║     3. Предсказание траектории (ML - линейная экстраполяция)                ║
║     4. Построение скелета                                                   ║
║     5. 5 обязательных CV картинок (нажмите D)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎮 УПРАВЛЕНИЕ:                                                              ║
║     ESC - выход                                                             ║
║     D   - создать 5 обязательных CV картинок                                ║
║     S   - сохранить скриншот                                                ║
║     H   - показать справку                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    cap = cv2.VideoCapture('data/v.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_color.mp4', fourcc, 20.0, (600,600))
    out1 = cv2.VideoWriter('output_skel.mp4', fourcc, 20.0, (600,600))
    out2 = cv2.VideoWriter('output_point.mp4', fourcc, 20.0, (600,600))
    out3 = cv2.VideoWriter('output_tr.mp4', fourcc, 20.0, (600,600))
    out4 = cv2.VideoWriter('output_concat.mp4', fourcc, 20.0, (1200,1200))
    if not cap.isOpened():
        print("❌ ОШИБКА: Не удалось открыть камеру!")
        return

    init_trackbars()
    tracker = Tracker()

   

    frame_count = 0
    fps = 0
    fps_start = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Детекция
        detected, masks = detect_all(hsv, None)

        # Трекинг
        tracks = tracker.update(detected)

        # Визуализация
        result = frame.copy()
        result = draw_trajectories(result, tracks)  # История движения
        result = draw_predictions(result,tracks)  # Предсказания
        result = draw_current_markers(result, detected)  # Текущие маркеры
        result = draw_skeleton(result, detected)  # Скелет

        # FPS
        if frame_count % 30 == 0:
            fps_end = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (fps_end - fps_start)
            fps_start = fps_end

        # Информация
        info = f"FPS: {fps:.1f} | Blue: {len(detected['blue'])}/3 | Green: {len(detected['green'])}/3 | Yellow: {len(detected['yellow'])}/8"
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, "D: CV images | S: Screenshot | ESC: Exit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        

        # Маски
        h, w = frame.shape[:2]
        mask_blue = cv2.cvtColor(masks['blue'], cv2.COLOR_GRAY2BGR)
        mask_green = cv2.cvtColor(masks['green'], cv2.COLOR_GRAY2BGR)
        mask_yellow = cv2.cvtColor(masks['yellow'], cv2.COLOR_GRAY2BGR)

        mask_blue[masks['blue'] > 0] = (255, 0, 0)
        mask_green[masks['green'] > 0] = (0, 255, 0)
        mask_yellow[masks['yellow'] > 0] = (0, 255, 255)

        # Показываем окна
        new = np.zeros(result.shape)
        new = draw_skeleton(new,detected)
        

        cv2.imshow('Skeleton & Tracking', cv2.resize(result,(600,600)))
        out.write(cv2.resize(result,(600,600)))

        #cv2.imshow('Color Bars',cv2.resize(color_bars,(600,600)))

        cv2.imshow('Masks', cv2.resize(mask_blue + mask_green + mask_yellow,(600,600)))
        out2.write(cv2.resize(mask_blue + mask_green + mask_yellow,(600,600)))

        cv2.imshow('skelet',cv2.resize(new,(600,600)))
        out1.write(cv2.resize(new,(600,600)))

        cv2.imshow('traeckt',cv2.resize(draw_predictions(np.zeros(result.shape),tracks),(600,600)))
        out3.write(cv2.resize(draw_predictions(np.zeros(result.shape),tracks),(600,600)))

        
        masks_row2 = np.hstack([cv2.resize(new,(600,600)),cv2.resize(draw_predictions(np.zeros(result.shape),tracks),(600,600))])
        masks_display = np.hstack([cv2.resize(mask_blue + mask_green + mask_yellow,(600,600)) , masks_row2])
        
        cv2.imshow("concated",cv2.resize(masks_display,(1200,1200)))
        out4.write(cv2.resize(masks_display,(1200,1200)))

 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'screenshot_{timestamp}.png', result)
            print(f"📸 Скриншот сохранен: screenshot_{timestamp}.png")
        elif key == ord('h'):
            print("""
=== СПРАВКА ===
ESC - выход
D   - создать 5 обязательных CV картинок
S   - сохранить скриншот
H   - показать справку

Настройка цветов:
- Двигайте ползунки H Min/Max для каждого цвета
- Смотрите на окно Masks - там белым видны маркеры
===============
""")
    out.release()
    out1.release()
    out2.release()
    out3.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Программа завершена. Результаты сохранены в папке проекта.")



if __name__ == "__main__":
    
    
    main()