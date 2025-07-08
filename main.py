import cv2
import sys
import os
from collections import deque
import numpy as np
from enum import Enum, auto
from ultralytics import YOLO

class Posture(Enum):
    STANDING = auto()
    SITTING = auto()
    LYING = auto()
    UNKNOWN = auto()

class Activity(Enum):
    INACTIVE = auto()
    CARRYING_ONE_HAND = auto()
    CARRYING_TWO_HANDS = auto()
    WAVING_HANDS = auto()
    WORKING_ON_COMPUTER = auto()
    EXERCISING = auto()

HISTORY_LEN = 15
WAVE_THRESHOLD = 0.5

pose_model = YOLO('yolov8n-pose.pt')

def main():
    source = select_source()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть источник {source}")
        return

    activity_history = deque(maxlen=HISTORY_LEN)
    wrist_pos_history = deque(maxlen=10)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    file_path = "data/output"
    file_nam = 1
    while os.path.isfile(file_path + f"{str(file_nam)}.mp4"):
        file_nam+= 1
    file_path+= str(file_nam) + ".mp4"

    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    print("Обработка видео... (Нажмите q для выхода)")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = pose_model(frame, verbose=False)
            current_activities = set()

            if results[0].keypoints is not None:
                for person in results[0].keypoints:
                    keypoints = person.xy[0].cpu().numpy()

                    # Определение позы
                    posture = classify_posture(keypoints)

                    # Определение активностей
                    activities = detect_activities(keypoints, wrist_pos_history)
                    current_activities.update(activities)

                    # Визуализация
                    draw_skeleton(frame, keypoints)
                    display_info(frame, posture, activities)

            # Анализ временных паттернов
            activity_history.append(current_activities)
            final_activities = analyze_activity_patterns(activity_history)

            cv2.imshow('Activity Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Готово! Результат сохранён в {file_path}")


def select_source():
    print("Выберите источник видео:")
    print("1 - Веб-камера")
    print("2 - Видеофайл")
    choice = input("Введите номер (1/2): ")

    if choice == '1':
        return 0
    elif choice == '2':
        path = input("Путь к видеофайлу: ")
        return path
    else:
        print("Неверный ввод, используется веб-камера по умолчанию")
        return 0


def classify_posture(keypoints):
    if len(keypoints) < 17:
        return Posture.UNKNOWN

    nose = keypoints[0]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    hip_y = (left_hip[1] + right_hip[1]) / 2
    knee_y = (left_knee[1] + right_knee[1]) / 2

    if abs(hip_y - knee_y) < 30:
        return Posture.LYING
    elif nose[1] - hip_y < 100:
        return Posture.SITTING
    return Posture.STANDING


def detect_activities(keypoints, wrist_history):
    activities = set()

    if len(keypoints) < 17:
        return {Activity.INACTIVE}

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # 1. Держит предметы
    hands_raised = check_hands_raised(left_wrist, right_wrist, left_elbow, right_elbow)
    if hands_raised == 2:
        activities.add(Activity.CARRYING_TWO_HANDS)
    elif hands_raised == 1:
        activities.add(Activity.CARRYING_ONE_HAND)

    # 2. Машет руками
    wrist_history.append((left_wrist, right_wrist))
    if check_waving(wrist_history):
        activities.add(Activity.WAVING_HANDS)

    # 3. Работает за компьютером
    if check_computer_work(left_wrist, right_wrist, left_elbow, right_elbow):
        activities.add(Activity.WORKING_ON_COMPUTER)

    # 4. Делает упражнения
    if check_exercising(keypoints):
        activities.add(Activity.EXERCISING)

    return activities if activities else {Activity.INACTIVE}


def check_hands_raised(lw, rw, le, re):
    left_raised = lw[1] < le[1] - 20
    right_raised = rw[1] < re[1] - 20
    return left_raised + right_raised


def check_waving(history):
    if len(history) < 5:
        return False

    horizontal_movement = 0
    for i in range(1, len(history)):
        (lw_prev, rw_prev), (lw_curr, rw_curr) = history[i - 1], history[i]
        horizontal_movement += abs(lw_curr[0] - lw_prev[0]) + abs(rw_curr[0] - rw_prev[0])

    return horizontal_movement / len(history) > WAVE_THRESHOLD


def check_computer_work(lw, rw, le, re):
    # Руки согнуты перед телом
    hands_in_front = (lw[0] > le[0] and rw[0] < re[0])
    hands_at_mid_level = (le[1] < re[1] + 50 and le[1] > re[1] - 50)
    return hands_in_front and hands_at_mid_level


def check_exercising(keypoints):
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    hip_knee_ratio = abs(left_hip[1] - left_knee[1]) / abs(left_hip[0] - left_knee[0])
    return hip_knee_ratio < 0.5  # Ноги согнуты


def analyze_activity_patterns(history):
    if not history:
        return {Activity.INACTIVE}

    flat_history = [act for acts in history for act in acts]
    if not flat_history:
        return {Activity.INACTIVE}

    return {max(set(flat_history), key=flat_history.count)}


def draw_skeleton(frame, keypoints):
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
        (6, 8), (7, 9), (8, 10), (11, 12), (11, 13),
        (12, 14), (13, 15), (14, 16)
    ]

    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            cv2.line(frame, tuple(map(int, keypoints[i])),
                     tuple(map(int, keypoints[j])), (0, 255, 0), 2)


def display_info(frame, posture, activities):
    cv2.putText(frame, f"Posture: {posture.name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for i, activity in enumerate(activities, start=2):
        cv2.putText(frame, f"Activity: {activity.name}", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

if __name__ == "__main__":
    main()