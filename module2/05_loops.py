# 05_loops.py
# Тема 5: Циклы (for / while)
# Запуск: python 05_loops.py

# Цикл - это повторение одного и того же кода несколько раз.

# for используется, когда известно количество повторений.
# range(n) создает последовательность чисел от 0 до n-1.

for i in range(5):
    print("i =", i)


# range(start, stop, step)
for i in range(2, 10, 2):
    print("even i =", i)


# Частый пример: подсчет суммы
total = 0
for i in range(1, 6):
    total += i
print("Сумма чисел от 1 до 5:", total)


# while используется, когда повторяем, пока условие True.
count = 0

while count < 3:
    print("count =", count)
    count += 1


# break - немедленно завершает цикл
for i in range(1, 10):
    if i == 6:
        break
    print("before break:", i)


# continue - пропускает текущую итерацию
for i in range(1, 10):
    if i == 4:
        continue
    print("after continue:", i)


# Пример с while и input()
# Раскомментируй для интерактива:
# while True:
#     text = input("Введите 'stop' для выхода: ").strip().lower()
#     if text == "stop":
#         break
#     print("Вы ввели:", text)

