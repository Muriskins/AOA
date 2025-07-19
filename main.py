import cv2
import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Отключить oneDNN-оптимизаций

import keras
import numpy as np
import sqlite3
import tensorflow as tf
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


def main():
    # Выбор источника
    source = select_source()

    # Загрузка обученной модели
    model = tf.keras.models.load_model("model.keras")

    # Загрузка классов из базы данных
    with sqlite3.connect("states.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code FROM states")
        classes = [row[0] for row in cursor.fetchall()]
    idx2class = {idx: cls for idx, cls in enumerate(classes)}

    # Инициализация детектора людей (HOG)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Инициализация MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:

        # Инициация видеопотока
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть источник {source}")
            return

        # Настройка видеопотока
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_path = "data/output"
        file_nam = 1
        while os.path.isfile(file_path + str(file_nam) + ".mp4"):
            file_nam += 1
        file_path += str(file_nam) + ".mp4"
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        print("Обработка видео... (Нажмите q для выхода)")

        # Основной цикл обработки
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Детекция людей с помощью HOG
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.05)

                # Обработка каждого обнаруженного человека
                for i, (x, y, w, h) in enumerate(boxes):
                    # Увеличение области обнаружения
                    padding = 30
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    # Вырезаем область с человеком
                    person_roi = frame[y1:y2, x1:x2]

                    # Пропускаем слишком маленькие области
                    if person_roi.size == 0:
                        continue

                    # Конвертация в RGB для MediaPipe
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    person_roi_rgb.flags.writeable = False

                    # Детекция позы для текущего человека
                    results = pose.process(person_roi_rgb)

                    if results.pose_landmarks:
                        # Извлечение ключевых точек
                        landmarks = results.pose_landmarks.landmark
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

                        # Подготовка изображения для модели
                        img_for_model = cv2.resize(person_roi, (224, 224))
                        img_for_model = img_for_model.astype(np.float32) / 255.0
                        img_for_model = np.expand_dims(img_for_model, axis=0)

                        # Предсказание класса
                        predictions = model.predict(
                            [img_for_model, np.expand_dims(keypoints, axis=0)],
                            verbose=0
                        )
                        class_idx = np.argmax(predictions[0])
                        class_label = idx2class[class_idx]
                        confidence = np.max(predictions[0])

                        # Создаем новый объект pose_landmarks с масштабированными координатами
                        pose_landmarks_scaled = landmark_pb2.NormalizedLandmarkList()
                        for landmark in results.pose_landmarks.landmark:
                            new_landmark = pose_landmarks_scaled.landmark.add()
                            new_landmark.x = (landmark.x * person_roi.shape[1] + x1) / frame.shape[1]
                            new_landmark.y = (landmark.y * person_roi.shape[0] + y1) / frame.shape[0]
                            new_landmark.z = landmark.z
                            new_landmark.visibility = landmark.visibility

                        # Отрисовка скелета на исходном изображении
                        mp_drawing.draw_landmarks(
                            frame,
                            pose_landmarks_scaled,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )

                        # Отрисовка bounding box и подписи
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_label} ({confidence:.2f})",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Сохранение и отображение кадра
                out.write(frame)
                cv2.imshow('Video Processing', frame)

                # Выход по клавише Q
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
    choice = input("Введите номер (1/2/ft): ")

    if choice == '1':
        return 0
    elif choice == '2':
        path = input("Путь к видеофайлу: ")
        return path
    elif choice == 'ft':  # Первый тест
        path = 'data/orig_video/1.mp4'
        return path
    else:
        print("Неверный ввод, используется веб-камера по умолчанию")
        return 0


if __name__ == "__main__":
    main()