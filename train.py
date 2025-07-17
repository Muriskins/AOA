import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import deque
import joblib


def prepare_dataset(csv_path, sequence_length=30, test_size=0.2):
    """
    Подготавливает данные для обучения модели
    :param csv_path: путь к CSV-файлу с размеченными данными
    :param sequence_length: длина последовательности для LSTM
    :param test_size: доля тестовых данных
    :return: кортеж с тренировочными и тестовыми данными
    """
    # Загрузка данных
    df = pd.read_csv(csv_path)

    # Фильтрация неразмеченных данных
    df = df[df['action'] != '']

    # Кодирование меток действий
    le = LabelEncoder()
    df['action_label'] = le.fit_transform(df['action'])

    # Список колонок с ключевыми точками
    kp_cols = [col for col in df.columns if col.startswith('kp')]

    # Нормализация данных
    scaler = StandardScaler()
    df[kp_cols] = scaler.fit_transform(df[kp_cols])

    # Создание последовательностей
    sequences = []
    labels = []
    track_ids = df['track_id'].unique()

    for track_id in track_ids:
        track_data = df[df['track_id'] == track_id]
        # Сортируем по кадрам
        track_data = track_data.sort_values('frame')

        # Создаем последовательности заданной длины
        for i in range(len(track_data) - sequence_length):
            seq = track_data.iloc[i:i + sequence_length][kp_cols].values
            # Берем последнее действие в последовательности как метку
            label = track_data.iloc[i + sequence_length - 1]['action_label']
            sequences.append(seq)
            labels.append(label)

    # Преобразование в numpy массивы
    X = np.array(sequences)
    y = np.array(labels)

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Сохранение препроцессоров
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    return X_train, X_test, y_train, y_test, le.classes_


def create_action_model(input_shape, num_classes):
    """
    Создает модель классификации действий
    :param input_shape: форма входных данных (timesteps, features)
    :param num_classes: количество классов действий
    :return: модель Keras
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=input_shape),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_model():
    # Подготовка данных
    X_train, X_test, y_train, y_test, classes = prepare_dataset(
        'labeled_poses.csv',
        sequence_length=30
    )

    print(f"Тренировочные данные: {X_train.shape}")
    print(f"Тестовые данные: {X_test.shape}")
    print(f"Классы: {classes}")

    # Создание модели
    model = create_action_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=len(classes)
    )

    # Коллбэки для обучения
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5)
    ]

    # Обучение модели
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    # Сохранение модели
    model.save('action_classifier.h5')
    print(f"Модель сохранена как action_classifier.h5")

    # Оценка модели
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Тестовая точность: {test_acc:.4f}")


if __name__ == "__main__":
    train_model()