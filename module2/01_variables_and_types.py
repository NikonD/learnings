# 01_variables_and_types.py
# Тема 1: Переменные и типы данных (строго, коротко, с примерами)
# Запуск: python 01_variables_and_types.py

# Переменная -  это сслылка на область в памяти, которое хранит значение.
# В Python тип определяется автоматически по значению.

# int - целые числа (без точки)
a = 10
b = -3

# float - числа с дробной частью (с точкой)
pi = 3.14
temp = -0.5

# str - строки (текст в кавычках)
name = "Nikita"
city = 'Petropavlovsk'

# bool - логика: True / False
is_ok = True
has_errors = False

print("a =", a, "| type:", type(a))
print("pi =", pi, "| type:", type(pi))
print("name =", name, "| type:", type(name))
print("is_ok =", is_ok, "| type:", type(is_ok))

# Важно: input() всегда возвращает строку (str)
# Раскомментируй и проверь:
# x = input("Введите что-то: ")
# print("x =", x, "| type:", type(x))

# Преобразование типов (кастинг)
# int("12") -> 12
# float("12.5") -> 12.5
# str(100) -> "100"
# bool(0) -> False, bool(1) -> True

print("int('12') =", int("12"))
print("float('12.5') =", float("12.5"))
print("str(100) =", str(100))
print("bool(0) =", bool(0), "| bool(10) =", bool(10))

# Частая ошибка новичков:
# int('12.5') вызовет ошибку ValueError, потому что это не целое число.
# Правильно: сначала float(), потом int() при необходимости.
print("int(float('12.5')) =", int(float("12.5")))

