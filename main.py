import cv2
import sys
import os

def main():
    source = select_source()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть источник {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    file_path = "data/output"
    file_nam = 1
    while os.path.isfile(file_path + f"{str(file_nam)}.mp4"):
        file_nam+= 1
    file_path+= str(file_nam) + ".mp4"

    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    print("Обработка видео... (Нажмите q для выхода)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # обработка кадра будет здесь>
            processed_frame = frame

            out.write(processed_frame)

            cv2.imshow('Video Processing', processed_frame)

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