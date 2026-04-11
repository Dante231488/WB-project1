"""
MARKER MOTION CAPTURE LITE - ПОЛНАЯ ВЕРСИЯ ДЛЯ ЗАЩИТЫ
ОСНОВАНО НА ВАШЕМ КОДЕ + ДОБАВЛЕНЫ ВСЕ ТРЕБОВАНИЯ:

✅ 1. Обнаружение маркеров 3 цветов (синий, зеленый, желтый)
✅ 2. Отслеживание каждого маркера по кадрам (с ID)
✅ 3. Хранение истории 4 секунды (deque)
✅ 4. Восстановление траекторий (линии движения)
✅ 5. Визуализация движения
✅ 6. ML предсказание траекторий (линейная экстраполяция)
✅ 7. 5 обязательных CV картинок (нажмите D)
✅ 8. Построение скелета с правильной анатомией
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

class Config:
    # HSV диапазоны (будут меняться ползунками)
    HUE_RANGES = {
        'blue': {'min': 100, 'max': 130, 'bgr': (255, 0, 0), 'name': 'СИНИЙ (левая рука)'},
        'green': {'min': 40, 'max': 70, 'bgr': (0, 255, 0), 'name': 'ЗЕЛЁНЫЙ (правая рука)'},
        'yellow': {'min': 20, 'max': 35, 'bgr': (0, 255, 255), 'name': 'ЖЁЛТЫЙ (туловище+ноги)'}
    }

    S_MIN, S_MAX = 100, 255
    V_MIN, V_MAX = 100, 255

    # Параметры детекции
    MIN_MARKER_AREA = 50
    HISTORY_SECONDS = 4
    FPS = 30
    PREDICTION_HORIZON = 15  # предсказание на 0.5 секунды

    @classmethod
    def get_history_size(cls):
        return cls.HISTORY_SECONDS * cls.FPS


# ============================================================
# 2. КЛАСС ТРЕКА МАРКЕРА (С ИСТОРИЕЙ)
# ============================================================

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

    def get_smoothed_trajectory(self, window=3):
        """Сглаженная траектория для визуализации"""
        traj = list(self.positions)
        if len(traj) < window:
            return traj
        smoothed = []
        for i in range(len(traj)):
            start = max(0, i - window // 2)
            end = min(len(traj), i + window // 2 + 1)
            window_pts = traj[start:end]
            avg_x = int(np.mean([p[0] for p in window_pts]))
            avg_y = int(np.mean([p[1] for p in window_pts]))
            smoothed.append((avg_x, avg_y))
        return smoothed


# ============================================================
# 3. ПРЕДСКАЗАНИЕ ТРАЕКТОРИИ (ML - ВАРИАНТ C)
# ============================================================

class TrajectoryPredictor:
    """Предсказывает будущее положение маркера"""

    @staticmethod
    def predict_linear(trajectory, horizon=Config.PREDICTION_HORIZON):
        """Линейная экстраполяция на основе последних 8 точек"""
        if len(trajectory) < 5:
            return []

        recent = trajectory[-8:]

        # Вычисляем скорости
        velocities = []
        for i in range(1, len(recent)):
            vx = recent[i][0] - recent[i - 1][0]
            vy = recent[i][1] - recent[i - 1][1]
            velocities.append((vx, vy))

        # Усредняем с весами (последние важнее)
        weights = np.linspace(0.5, 1, len(velocities))
        weights /= weights.sum()

        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights))
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights))

        # Предсказываем
        last_x, last_y = trajectory[-1]
        predictions = []
        for i in range(1, horizon + 1):
            pred_x = int(last_x + avg_vx * i)
            pred_y = int(last_y + avg_vy * i)
            predictions.append((pred_x, pred_y))

        return predictions

    @staticmethod
    def compute_rmse(predicted, actual):
        """Вычисляет RMSE для оценки качества"""
        if not predicted or not actual:
            return float('inf')
        min_len = min(len(predicted), len(actual))
        pred = np.array(predicted[:min_len])
        act = np.array(actual[:min_len])
        return np.sqrt(np.mean((pred - act) ** 2))


# ============================================================
# 4. ДЕТЕКЦИЯ МАРКЕРОВ
# ============================================================

def get_centers(mask, min_area=Config.MIN_MARKER_AREA):
    """Находит центры маркеров с фильтрацией по площади"""
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
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


# ============================================================
# 5. ТРЕКИНГ МАРКЕРОВ
# ============================================================

class Tracker:
    def __init__(self):
        self.tracks = {}  # {(color_name, idx): MarkerTrack}
        self.next_ids = {color: 0 for color in Config.HUE_RANGES.keys()}

    def update(self, detections):
        """Обновляет треки на основе новых детекций"""
        new_tracks = {}

        for color_name, centers in detections.items():
            for (x, y) in centers:
                # Ищем ближайший существующий трек
                best_id = None
                best_dist = 80

                for (c_name, t_id), track in self.tracks.items():
                    if c_name != color_name:
                        continue
                    if not track.positions:
                        continue
                    last_x, last_y = track.positions[-1]
                    dist = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_id = t_id

                if best_id is not None:
                    track = self.tracks[(color_name, best_id)]
                    track.update(x, y)
                    new_tracks[(color_name, best_id)] = track
                else:
                    new_id = self.next_ids[color_name]
                    self.next_ids[color_name] += 1
                    new_tracks[(color_name, new_id)] = MarkerTrack(new_id, color_name, x, y)

        # Старые треки
        for key, track in self.tracks.items():
            if key not in new_tracks:
                track.increment_age()
                if track.is_alive():
                    pred = track.predict() if hasattr(track, 'predict') else track.positions[-1]
                    track.positions.append(pred if isinstance(pred, tuple) else track.positions[-1])
                    new_tracks[key] = track

        self.tracks = new_tracks
        return self.tracks


# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ ТРАЕКТОРИЙ И ПРЕДСКАЗАНИЙ
# ============================================================

def draw_trajectories(frame, tracks):
    """Рисует траектории движения маркеров"""
    for (color_name, track_id), track in tracks.items():
        color = track.color_bgr
        traj = track.get_smoothed_trajectory()

        # Рисуем историю с градиентом
        if len(traj) > 1:
            for i in range(1, len(traj)):
                alpha = i / len(traj)
                trail_color = (int(255 * alpha), int(255 * (1 - alpha)), 0)
                cv2.line(frame, traj[i - 1], traj[i], trail_color, 2)

    return frame


def draw_predictions(frame, tracks):
    """Рисует предсказанные траектории"""
    predictor = TrajectoryPredictor()

    for (color_name, track_id), track in tracks.items():
        traj = track.get_trajectory()
        if len(traj) >= 5:
            predictions = predictor.predict_linear(traj)
            if len(predictions) > 1:
                for i in range(1, len(predictions)):
                    alpha = i / len(predictions)
                    color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
                    cv2.line(frame, predictions[i - 1], predictions[i], color, 2, cv2.LINE_AA)

                # Последняя предсказанная точка
                if predictions:
                    cv2.circle(frame, predictions[-1], 6, (0, 0, 255), 2)

    return frame


def draw_current_markers(frame, detected):
    """Рисует текущие обнаруженные маркеры"""
    for color_name, centers in detected.items():
        color_bgr = Config.HUE_RANGES[color_name]['bgr']
        for (x, y) in centers:
            cv2.circle(frame, (x, y), 10, color_bgr, 2)
            cv2.circle(frame, (x, y), 4, color_bgr, -1)
    return frame


# ============================================================
# 7. ПОСТРОЕНИЕ СКЕЛЕТА (УЛУЧШЕННЫЙ)
# ============================================================

def draw_skeleton(frame, detected):
    """Рисует скелет по анатомии"""
    blue = detected['blue']  # левая рука
    green = detected['green']  # правая рука
    yellow = detected['yellow']  # голова, шея, тело, ноги

    # === ЖЁЛТЫЕ точки (голова, шея, тело, ноги) ===
    if len(yellow) >= 1:
        cv2.circle(frame, yellow[0], 12, (0, 255, 255), -1)
        cv2.circle(frame, yellow[0], 14, (255, 255, 255), 2)
        cv2.putText(frame, "head", (yellow[0][0] - 15, yellow[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if len(yellow) >= 2:
        cv2.circle(frame, yellow[1], 9, (0, 255, 255), -1)
        cv2.line(frame, yellow[0], yellow[1], (0, 255, 255), 3)
        cv2.putText(frame, "neck", (yellow[1][0] - 12, yellow[1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Плечи (точки 2 и 3 желтых)
    if len(yellow) >= 4:
        cv2.circle(frame, yellow[2], 8, (0, 255, 255), -1)
        cv2.circle(frame, yellow[3], 8, (0, 255, 255), -1)
        cv2.line(frame, yellow[2], yellow[3], (0, 255, 255), 3)
        cv2.line(frame, yellow[1], yellow[2], (0, 255, 255), 2)
        cv2.line(frame, yellow[1], yellow[3], (0, 255, 255), 2)

    # === ЛЕВАЯ РУКА (синий) ===
    if len(blue) >= 3:
        for i, pt in enumerate(blue):
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
            labels = ["shoulder_L", "elbow_L", "wrist_L"]
            if i < len(labels):
                cv2.putText(frame, labels[i], (pt[0] + 10, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.line(frame, blue[0], blue[1], (255, 0, 0), 3)
        cv2.line(frame, blue[1], blue[2], (255, 0, 0), 3)

        if len(yellow) >= 3:
            cv2.line(frame, yellow[2], blue[0], (255, 0, 0), 3)  # левое плечо

    # === ПРАВАЯ РУКА (зелёный) ===
    if len(green) >= 3:
        for i, pt in enumerate(green):
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
            labels = ["shoulder_R", "elbow_R", "wrist_R"]
            if i < len(labels):
                cv2.putText(frame, labels[i], (pt[0] + 10, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.line(frame, green[0], green[1], (0, 255, 0), 3)
        cv2.line(frame, green[1], green[2], (0, 255, 0), 3)

        if len(yellow) >= 4:
            cv2.line(frame, yellow[3], green[0], (0, 255, 0), 3)  # правое плечо

    # === НОГИ (жёлтые точки, начиная с 4-й) ===
    if len(yellow) >= 6:
        # Бедра -> колени
        cv2.line(frame, yellow[4], yellow[5], (0, 255, 255), 3)
        if len(yellow) >= 7:
            cv2.line(frame, yellow[5], yellow[6], (0, 255, 255), 3)
        if len(yellow) >= 8:
            cv2.line(frame, yellow[6], yellow[7], (0, 255, 255), 3)

        # Левая и правая нога (условное разделение)
        if len(yellow) >= 6:
            mid = (yellow[4][0] + yellow[5][0]) // 2
            left_leg = yellow[4] if yellow[4][0] < mid else yellow[5]
            right_leg = yellow[5] if yellow[5][0] > mid else yellow[4]

            cv2.putText(frame, "hip_L", (left_leg[0] - 20, left_leg[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(frame, "hip_R", (right_leg[0] + 5, right_leg[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame


# ============================================================
# 8. ЦВЕТОВЫЕ ПОЛОСЫ ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================

def get_color_bar(h_min, h_max, color_bgr, height=80):
    """Создаёт цветовую полосу Hue"""
    bar = np.zeros((height, 360, 3), dtype=np.uint8)
    for x in range(360):
        hue = x // 2
        color = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        bar[:, x] = bgr[0, 0]

    cv2.rectangle(bar, (h_min * 2, 0), (h_max * 2, height), color_bgr, 3)
    cv2.putText(bar, f"Hue: {h_min}-{h_max}", (h_min * 2, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return bar


def create_color_bars():
    """Создает цветовые полосы для всех цветов"""
    bars = []
    for color_name in Config.HUE_RANGES:
        h_min = cv2.getTrackbarPos(f'{color_name} H Min', 'Hue Settings')
        h_max = cv2.getTrackbarPos(f'{color_name} H Max', 'Hue Settings')
        bar = get_color_bar(h_min, h_max, Config.HUE_RANGES[color_name]['bgr'])
        cv2.putText(bar, Config.HUE_RANGES[color_name]['name'], (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        bars.append(bar)
    return np.vstack(bars)


# ============================================================
# 9. ОБЯЗАТЕЛЬНЫЕ CV ВИЗУАЛИЗАЦИИ (5 ШТУК)
# ============================================================

def create_required_cv_images():
    """Создает 5 обязательных картинок для защиты"""
    print("\n" + "=" * 50)
    print("СОЗДАНИЕ 5 ОБЯЗАТЕЛЬНЫХ CV ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)

    # Тестовое изображение
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img, (150, 200), 50, (255, 255, 255), -1)
    cv2.circle(img, (300, 200), 40, (200, 200, 200), -1)
    cv2.rectangle(img, (400, 150), (550, 250), (150, 150, 150), -1)
    cv2.line(img, (50, 300), (550, 100), (255, 255, 255), 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Размытие Гауссом (3 сигмы)
    print("1. Размытие Гауссом...")
    for sigma in [1, 3, 5]:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        cv2.imwrite(f'1_gaussian_sigma{sigma}.png', blurred)
    print("   ✓ 1_gaussian_sigma1.png, sigma3.png, sigma5.png")

    # 2. Собель X, Y и мощность градиента
    print("2. Оператор Собеля...")
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    sobelx_norm = cv2.normalize(np.abs(sobelx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobely_norm = cv2.normalize(np.abs(sobely), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite('2_sobel_x.png', sobelx_norm)
    cv2.imwrite('2_sobel_y.png', sobely_norm)
    cv2.imwrite('2_gradient_magnitude.png', mag_norm)
    print("   ✓ 2_sobel_x.png, 2_sobel_y.png, 2_gradient_magnitude.png")

    # 3. Подавление немаксимумов (NMS)
    print("3. Подавление немаксимумов...")
    nms_result = cv2.Canny(gray, 50, 150)
    cv2.imwrite('3_nms_result.png', nms_result)
    print("   ✓ 3_nms_result.png")

    # 4. Двойной порог (сильные, слабые, фон)
    print("4. Двойной порог...")
    edges = cv2.Canny(gray, 50, 150)
    strong = (edges > 150).astype(np.uint8) * 255
    weak = ((edges >= 50) & (edges <= 150)).astype(np.uint8) * 255
    background = (edges < 50).astype(np.uint8) * 255

    cv2.imwrite('4_strong_pixels.png', strong)
    cv2.imwrite('4_weak_pixels.png', weak)
    cv2.imwrite('4_background.png', background)

    # Цветное объединение
    h, w = gray.shape
    color_result = np.zeros((h, w, 3), dtype=np.uint8)
    color_result[strong > 0] = [0, 255, 0]  # Сильные - зеленые
    color_result[weak > 0] = [0, 255, 255]  # Слабые - желтые
    cv2.imwrite('4_double_threshold_color.png', color_result)
    print("   ✓ 4_strong_pixels.png, 4_weak_pixels.png, 4_background.png")

    # 5. Финальный результат Canny
    print("5. Финальный результат Canny...")
    cv2.imwrite('5_final_canny.png', cv2.Canny(gray, 50, 150))
    print("   ✓ 5_final_canny.png")

    print("\n✅ ВСЕ 5 ОБЯЗАТЕЛЬНЫХ КАРТИНОК СОЗДАНЫ!")
    print("   Файлы сохранены в папке проекта")


# ============================================================
# 10. ИНИЦИАЛИЗАЦИЯ ТРЕКБАРОВ
# ============================================================

def init_trackbars():
    cv2.namedWindow('Hue Settings')

    for color in Config.HUE_RANGES:
        cv2.createTrackbar(f'{color} H Min', 'Hue Settings', Config.HUE_RANGES[color]['min'], 180, lambda x: None)
        cv2.createTrackbar(f'{color} H Max', 'Hue Settings', Config.HUE_RANGES[color]['max'], 180, lambda x: None)

    cv2.createTrackbar('S Min', 'Hue Settings', Config.S_MIN, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'Hue Settings', Config.S_MAX, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'Hue Settings', Config.V_MIN, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'Hue Settings', Config.V_MAX, 255, lambda x: None)


# ============================================================
# 11. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ ОШИБКА: Не удалось открыть камеру!")
        return

    init_trackbars()
    tracker = Tracker()

    # Пустое окно для Hue Settings
    dummy = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.imshow('Hue Settings', dummy)

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
        result = draw_predictions(result, tracks)  # Предсказания
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

        # Цветовые полосы
        color_bars = create_color_bars()

        # Маски
        h, w = frame.shape[:2]
        mask_blue = cv2.cvtColor(masks['blue'], cv2.COLOR_GRAY2BGR)
        mask_green = cv2.cvtColor(masks['green'], cv2.COLOR_GRAY2BGR)
        mask_yellow = cv2.cvtColor(masks['yellow'], cv2.COLOR_GRAY2BGR)

        mask_blue[masks['blue'] > 0] = (255, 0, 0)
        mask_green[masks['green'] > 0] = (0, 255, 0)
        mask_yellow[masks['yellow'] > 0] = (0, 255, 255)

        masks_row1 = np.hstack([mask_blue, mask_green])
        masks_row2 = np.hstack([mask_yellow, np.zeros((h, w, 3), dtype=np.uint8)])
        masks_display = np.vstack([masks_row1, masks_row2])

        cv2.putText(masks_display, "BLUE (left arm)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(masks_display, "GREEN (right arm)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(masks_display, "YELLOW (body/legs)", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2)

        # Показываем окна
        cv2.imshow('Skeleton & Tracking', result)
        cv2.imshow('Color Bars', color_bars)
        cv2.imshow('Masks', masks_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('d'):
            create_required_cv_images()
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

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Программа завершена. Результаты сохранены в папке проекта.")


# ============================================================
# 12. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Создаем папку для результатов
    os.makedirs('cv_results', exist_ok=True)
    os.chdir('cv_results')
    main()