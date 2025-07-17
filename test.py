import cv2
import sys
import os
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

def main():
    source = select_source()

    # Загрузка модели YOLOv8
    model = YOLO("yolov8n-pose.pt")

    # Открытие видео файла
    cap = cv2.VideoCapture(source)

    # Настройка отображения и записи
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Определение пути записи видео файла
    file_path = "out_data/output"
    file_nam = 1
    while os.path.isfile(file_path + str(file_nam) + ".mp4"):
        file_nam += 1
    file_path += str(file_nam) + ".mp4"


    # Проверка успешного открытия видео
    if not cap.isOpened():
        print(f"Ошибка открытия {file_path}")
        exit()

    # Настройка VideoWriter для сохранения выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    # Создание словаря для хранения истории треков объектов
    track_history = defaultdict(lambda: [])

    # Цикл для обработки каждого кадра видео
    while cap.isOpened():
        # Считывание кадра из видео
        success, frame = cap.read()

        if not success:
            print("Конец видео")
            break

        # Применение YOLOv8 для отслеживания объектов на кадре, с сохранением треков между кадрами
        results = model.track(frame, persist=True)

        # Проверка на наличие объектов
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Получение координат боксов и идентификаторов треков
            boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
            track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков

            # Визуализация результатов на кадре
            annotated_frame = results[0].plot()

            # Отрисовка треков
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                track.append((float(x), float(y)))  # добавление координат центра объекта в историю
                if len(track) > 30:  # ограничение длины истории до 30 кадров
                    track.pop(0)

                # Рисование линий трека
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Отображение аннотированного кадра
            cv2.imshow("YOLOv11 Tracking", annotated_frame)
            out.write(annotated_frame)  # запись кадра в выходное видео
        else:
            # Если объекты не обнаружены, просто отображаем кадр
            cv2.imshow("YOLOv11 Tracking", frame)
            out.write(frame)  # запись кадра в выходное видео

        # Прерывание цикла при нажатии клавиши 'Esc'
        if cv2.waitKey(1) == 27:
            break


    # Освобождение видеозахвата и закрытие всех окон OpenCV
    cap.release()
    out.release()  # закрытие выходного видеофайла
    cv2.destroyAllWindows()


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


if __name__ == "__main__":
    main()