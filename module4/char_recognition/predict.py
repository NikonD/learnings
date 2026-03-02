"""
Предсказание цифры по изображению.
Картинка должна быть 28×28 пикселей, один символ (белый на чёрном или чёрный на белом, как в MNIST).
Если картинка другого размера — она приводится к 28×28.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "model.keras"

# Размер входа модели MNIST
IMG_SIZE = 28


def load_and_prepare_image(path: str | Path) -> np.ndarray:
    """Загружает изображение и приводит к формату модели: (1, 28, 28, 1), значения 0–1."""
    img = Image.open(path).convert("L")  # в оттенки серого
    arr = np.array(img)

    # Инвертируем, если фон светлый (в MNIST цифра белая на чёрном)
    if arr.mean() > 127:
        arr = 255 - arr

    img_small = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img_small, dtype="float32") / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)
    return arr


def predict_digit(image_path: str | Path) -> int:
    """Возвращает предсказанную цифру (0–9)."""
    if not MODEL_PATH.exists():
        print("Сначала обучите модель: python train.py")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    x = load_and_prepare_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    return int(np.argmax(probs))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <путь_к_картинке>")
        print("Пример: python predict.py digit.png")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)

    digit = predict_digit(path)
    print(f"Распознанная цифра: {digit}")
