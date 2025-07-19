import cv2
import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Отключить oneDNN-оптимизаций


import keras

import numpy as np
import sqlite3
import tensorflow as tf
import mediapipe as mp


def main():
    # Выбор источника
    source = select_source()

    # Загрузка обученной модели
    model = tf.keras.models.load_model("models/model.keras")

    # Загрузка классов из базы данных
    with sqlite3.connect("models/states.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code FROM states")
        classes = [row[0] for row in cursor.fetchall()]
    idx2class = {idx: cls for idx, cls in enumerate(classes)}

    # Инициализация MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Инициация машинного зрение
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть источник {source}")
        return

    # Его настройка
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    file_path = "data/output"
    file_nam = 1
    while os.path.isfile(file_path + str(file_nam) + ".mp4"):
        file_nam+= 1
    file_path+= str(file_nam) + ".mp4"

    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    print("Обработка видео... (Нажмите q для выхода)")

    # Основной цикл
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Конвертация в RGB (требование MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Детекция позы
            results = pose.process(frame_rgb)
            keypoints = np.zeros(33 * 3, dtype=np.float32)

            # Если поза обнаружена
            if results.pose_landmarks:
                # Отрисовка скелета
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

                # Извлечение ключевых точек
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

            # Подготовка данных для модели
            img_for_model = cv2.resize(frame, (224, 224))
            img_for_model = img_for_model.astype(np.float32) / 255.0
            img_for_model = np.expand_dims(img_for_model, axis=0)  # Добавляем размерность батча

            # Предсказание класса
            predictions = model.predict(
                [img_for_model, np.expand_dims(keypoints, axis=0)],
                verbose=0
            )
            class_idx = np.argmax(predictions[0])
            class_label = idx2class[class_idx]
            confidence = np.max(predictions[0])

            # Отображение результата на кадре
            cv2.putText(frame, f"{class_label} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
    elif choice == 'ft': # Первый тест
        path = 'data/orig_video/1.mp4'
        return path
    else:
        print("Неверный ввод, используется веб-камера по умолчанию")
        return 0


if __name__ == "__main__":
    main()