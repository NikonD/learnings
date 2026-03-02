"""
Обучение модели на датасете MNIST (рукописные цифры 0–9).
После обучения модель сохраняется в model.keras.
"""
from pathlib import Path

import tensorflow as tf

from model import build_model

# Папка проекта — сюда сохраняем модель
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "model.keras"

# Загрузка MNIST (скачивается при первом запуске)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормализация 0–255 -> 0–1 и добавление канала (28, 28) -> (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = build_model()
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
)

model.save(MODEL_PATH)
print(f"Модель сохранена: {MODEL_PATH}")
