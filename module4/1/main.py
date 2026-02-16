import numpy as np
import json

# --------------------------
# 1. Данные
# --------------------------
X = np.array([30, 40, 50, 60, 70])
y = np.array([95, 120, 145, 170, 195])

# --------------------------
# 2. Обучение модели
# --------------------------
a = 0.0
b = 0.0
lr = 0.0001

for _ in range(5000): # количество проходов
    y_pred = a * X + b
    error = y_pred - y

    da = (2/len(X)) * np.sum(error * X)
    db = (2/len(X)) * np.sum(error)

    a -= lr * da
    b -= lr * db

print("Обучено:")
print("a =", a)
print("b =", b)

# --------------------------
# 3. Сохранение модели
# --------------------------
model = {
    "a": float(a),
    "b": float(b)
}

with open("model_one_feature.json", "w") as f:
    json.dump(model, f)

print("Модель сохранена в model_one_feature.json")

# --------------------------
# 4. Загрузка модели
# --------------------------
with open("model_one_feature.json", "r") as f:
    loaded_model = json.load(f)

# --------------------------
# 5. Предсказание
# --------------------------
area = 55
predicted_price = loaded_model["a"] * area + loaded_model["b"]

print(f"Предсказанная цена для {area} м²:", predicted_price)
