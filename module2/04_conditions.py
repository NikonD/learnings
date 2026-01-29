# 04_conditions.py
# Тема 4: Условия (if / elif / else)
# Запуск: python 04_conditions.py

# Условие - это проверка, которая возвращает True или False.
# В зависимости от результата выполняется нужный блок кода.

# Операторы сравнения:
# == равно
# != не равно
# > больше
# < меньше
# >= больше или равно
# <= меньше или равно

a = 10
b = 5

print("a == b:", a == b)
print("a != b:", a != b)
print("a > b:", a > b)
print("a < b:", a < b)


# if - выполняется, если условие True
x = 7

if x > 0:
    print("Число положительное")


# if / else - выбор из двух вариантов
y = -3

if y >= 0:
    print("Неотрицательное число")
else:
    print("Отрицательное число")


# if / elif / else - несколько вариантов
score = 82

if score >= 90:
    grade = "Отлично"
elif score >= 70:
    grade = "Хорошо"
elif score >= 50:
    grade = "Удовлетворительно"
else:
    grade = "Неудовлетворительно"

print("Оценка:", grade)


# Логические операторы:
# and - и (оба условия True)
# or - или (хотя бы одно True)
# not - не (инверсия)

age = 19
has_ticket = True

if age >= 18 and has_ticket:
    print("Вход разрешен")
else:
    print("Вход запрещен")


is_admin = False

if not is_admin:
    print("Обычный пользователь")


# Условия часто используются вместе с input()
# Раскомментируй для интерактивного ввода:
# number = int(input("Введите число: "))
# if number % 2 == 0:
#     print("Четное")
# else:
#     print("Нечетное")

