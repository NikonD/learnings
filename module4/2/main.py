import numpy as np
import json

# --------------------------
# 1. Данные
# --------------------------
X = np.array([
    [30, 1980],
    [30, 2015],
    [50, 1985],
    [50, 2020],
    [70, 1990],
    [70, 2022]
])

y = np.array([80, 110, 130, 170, 170, 220])

# --------------------------
# 2. Обучение модели
# --------------------------
a = 0.0   # коэффициент для площади
c = 0.0   # коэффициент для года
b = 0.0   # смещение

lr = 1e-10

for _ in range(20000): # количество проходов
    y_pred = a * X[:,0] + c * X[:,1] + b
    error = y_pred - y

    da = (2/len(X)) * np.sum(error * X[:,0])
    dc = (2/len(X)) * np.sum(error * X[:,1])
    db = (2/len(X)) * np.sum(error)

    a -= lr * da
    c -= lr * dc
    b -= lr * db

print("Обучено:")
print("a (площадь) =", a)
print("c (год) =", c)
print("b =", b)

# --------------------------
# 3. Сохранение модели
# --------------------------
model = {
    "a_area": float(a),
    "c_year": float(c),
    "b": float(b)
}

with open("model_two_features.json", "w") as f:
    json.dump(model, f)

print("Модель сохранена в model_two_features.json")

# --------------------------
# 4. Загрузка модели
# --------------------------
with open("model_two_features.json", "r") as f:
    loaded_model = json.load(f)

# --------------------------
# 5. Предсказание
# --------------------------
area = 60
year = 2022

predicted_price = (
    loaded_model["a_area"] * area +
    loaded_model["c_year"] * year +
    loaded_model["b"]
)

print(f"Предсказанная цена для {area} м², {year} года:", predicted_price)
