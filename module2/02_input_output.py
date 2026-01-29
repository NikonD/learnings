# 02_input_output.py
# Тема 2: Ввод и вывод данных
# Запуск: python 02_input_output.py

# print() используется для вывода информации в консоль.
print("Hello, world")
print(10 + 5)

name = "Alex"
age = 20
print("Name:", name, "Age:", age)

# print() может управлять разделителем и окончанием строки
print("A", "B", "C", sep=", ")
print("Line 1", end=" | ")
print("Line 2")

print()  # пустая строка


# input() используется для ввода данных с клавиатуры.
# input() ВСЕГДА возвращает строку (str).

# Пример ввода строки:
# user_name = input("Enter your name: ")
# print("Hello,", user_name)

# Пример ввода числа:
# Чтобы работать с числами, нужно преобразование типа.
# number = int(input("Enter number: "))
# print("Number * 2 =", number * 2)

# Пример с float:
# price = float(input("Enter price: "))
# print("Final price:", price)

# Частая ошибка новичков:
# number = input("Enter number: ")
# print(number + 10)  # Ошибка: сложение строки и числа

# Правильно:
# number = int(input("Enter number: "))
# print(number + 10)

# input() можно сразу использовать внутри функций:
# print(int(input("A: ")) + int(input("B: ")))

