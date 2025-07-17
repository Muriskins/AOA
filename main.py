import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import torch
from mmengine import Config
from mmaction.apis import init_recognizer
from mmpose.apis import inference_topdown, init_model as init_pose_model
from mmpose.structures import merge_data_samples


def prepare_posec3d_input(keypoint_sequence):
    """Подготовка данных для PoseC3D в правильном формате"""
    num_frames = len(keypoint_sequence)
    num_joints = 17  # COCO keypoints

    # Инициализация массива данных: [N, C, T, V, M] = [1, 3, T, 17, 1]
    data = np.zeros((1, 3, num_frames, num_joints, 1), dtype=np.float32)

    # Заполнение данных
    for t, frame_data in enumerate(keypoint_sequence):
        keypoints = frame_data['keypoints']
        scores = frame_data['scores']

        for j in range(num_joints):
            # Координаты X, Y и confidence score
            data[0, 0, t, j, 0] = keypoints[j, 0]  # X координата
            data[0, 1, t, j, 0] = keypoints[j, 1]  # Y координата
            data[0, 2, t, j, 0] = scores[j]  # Confidence score

    return data


def main():
    source = select_source()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Загрузка модели YOLOv8 для детекции людей
    yolo_model = YOLO("yolov8n.pt").to(device)

    # 2. Инициализация модели MMPose для оценки позы
    pose_config = 'configs_pose/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    pose_checkpoint = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    # 3. Инициализация модели MMAction2 для распознавания действий
    action_config = 'configs_act/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
    action_checkpoint = 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth'
    action_model = init_recognizer(action_config, action_checkpoint, device=device)

    # Загрузка меток классов
    action_labels = load_action_labels()

    # Открытие видео
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Настройка пути для сохранения видео
    file_path = "out_data/output"
    file_num = 1
    while os.path.isfile(f"{file_path}{file_num}.mp4"):
        file_num += 1
    output_path = f"{file_path}{file_num}.mp4"

    if not cap.isOpened():
        print(f"Ошибка открытия {source}")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Структуры для хранения данных
    track_history = defaultdict(lambda: [])
    keypoint_history = defaultdict(lambda: [])
    action_history = defaultdict(lambda: [])
    last_prediction = {}
    last_seen = {}  # Добавлен словарь для отслеживания последнего появления
    frame_count = 0
    frame_interval = 10
    sequence_length = 30

    while cap.isOpened():
        frame_count += 1
        success, frame = cap.read()

        if not success:
            print("Конец видео")
            break

        # 1. Детекция объектов с помощью YOLOv8
        det_results = yolo_model.track(
            frame,
            persist=True,
            classes=[0],
            verbose=False,
            conf=0.5
        )

        annotated_frame = det_results[0].plot()

        if det_results[0].boxes is not None and det_results[0].boxes.id is not None:
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            track_ids = det_results[0].boxes.id.int().cpu().tolist()

            # Обновляем last_seen для всех обнаруженных треков
            for track_id in track_ids:
                last_seen[track_id] = frame_count

            pose_input_bboxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                pose_input_bboxes.append([x1, y1, w, h])

            # 2. Оценка позы с помощью MMPose
            pose_results = inference_topdown(
                pose_model,
                frame,
                pose_input_bboxes,
                bbox_format='xywh'
            )
            pose_results = merge_data_samples(pose_results)

            for i, track_id in enumerate(track_ids):
                if i < len(pose_results.pred_instances.keypoints):
                    keypoints = pose_results.pred_instances.keypoints[i]
                    keypoint_scores = pose_results.pred_instances.keypoint_scores[i]

                    if len(keypoint_history[track_id]) >= sequence_length:
                        keypoint_history[track_id].pop(0)
                    keypoint_history[track_id].append({
                        'keypoints': keypoints,
                        'scores': keypoint_scores
                    })

                    # 3. Распознавание действий
                    if frame_count % frame_interval == 0 and len(keypoint_history[track_id]) >= sequence_length:
                        input_data = prepare_posec3d_input(keypoint_history[track_id])
                        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(device)

                        with torch.no_grad():
                            predictions = action_model(input_tensor)

                        pred_scores = predictions[0].cpu().numpy()
                        pred_label = np.argmax(pred_scores)
                        pred_class = action_labels.get(str(pred_label + 1), "Unknown")
                        confidence = np.max(pred_scores)

                        if len(action_history[track_id]) >= 5:
                            action_history[track_id].pop(0)
                        action_history[track_id].append((pred_class, confidence))

                        action_counts = {}
                        for action, conf in action_history[track_id]:
                            action_counts[action] = action_counts.get(action, 0) + 1

                        if action_counts:
                            most_common = max(action_counts, key=action_counts.get)
                            last_prediction[track_id] = most_common

                    # Отображение действия
                    if track_id in last_prediction:
                        action_text = last_prediction[track_id]
                        x1, y1, x2, y2 = boxes[i]
                        cv2.putText(
                            annotated_frame,
                            action_text,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                    # Отрисовка трека
                    track = track_history[track_id]
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    track.append(center)
                    if len(track) > 20:
                        track.pop(0)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], False, (230, 230, 230), 2)

            out.write(annotated_frame)
            cv2.imshow("Action Recognition", annotated_frame)
        else:
            out.write(frame)
            cv2.imshow("Action Recognition", frame)

        # Очистка устаревших треков
        stale_tracks = [tid for tid in list(keypoint_history.keys()) if frame_count - last_seen.get(tid, 0) > 100]
        for tid in stale_tracks:
            del keypoint_history[tid]
            del action_history[tid]
            del last_prediction[tid]
            del last_seen[tid]

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def prepare_posec3d_input(keypoint_sequence):
    num_frames = len(keypoint_sequence)
    num_joints = 17

    data = np.zeros((3, num_frames, num_joints, 1), dtype=np.float32)

    for t, frame_data in enumerate(keypoint_sequence):
        keypoints = frame_data['keypoints']
        scores = frame_data['scores']

        for j in range(num_joints):
            data[0, t, j, 0] = keypoints[j, 0]
            data[1, t, j, 0] = keypoints[j, 1]
            data[2, t, j, 0] = scores[j]

    return data


def load_action_labels():
    return {
        "1": "drink", "2": "eat", "3": "brush_teeth",
        "8": "sit_down", "9": "stand_up", "23": "wave",
        "55": "hug", "59": "walk_to", "60": "walk_away"
    }


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
    elif choice == 'ft':
        path = "data/orig_video/1.mp4"
        return path
    else:
        print("Неверный ввод, используется веб-камера по умолчанию")
        return 0



if __name__ == "__main__":
    main()