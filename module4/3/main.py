import numpy as np

# -----------------------
# 1. Данные
# -----------------------

diameter = np.array([9.97, 9.99, 10.01, 10.03, 10.05])
tested = np.array([1000, 1000, 1000, 1000, 1000])
defects = np.array([5, 12, 45, 120, 300])

# центрируем
x = diameter - 10.0

# наблюдаемая доля брака
y_rate = defects / tested

# -----------------------
# 2. Модель
# -----------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = 0.0
b = 0.0
lr = 0.01

# -----------------------
# 3. Обучение (взвешенное)
# -----------------------

for _ in range(5000):
    z = w * x + b
    p = sigmoid(z)

    # градиенты с учётом количества деталей
    dw = np.mean(tested * (p - y_rate) * x)
    db = np.mean(tested * (p - y_rate))

    w -= lr * dw
    b -= lr * db

print("w =", w)
print("b =", b)

# -----------------------
# 4. Предсказание
# -----------------------

test_value = float(input("Диаметр: "))
test_value -= 10.0

prob = sigmoid(w * test_value + b)

print("Оценка вероятности брака:", prob)
print("Ожидаемый брак на 1000 деталей:", prob * 1000)