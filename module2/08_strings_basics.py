# 08_strings_basics.py
# Тема 8: Строки (str)
# Индексы и базовые методы: lower(), upper(), strip()
# Запуск: python 08_strings_basics.py

# str - это последовательность символов.
text = "Python"

print("text =", text)

# Индексы начинаются с 0
print("text[0] =", text[0])
print("text[1] =", text[1])
print("text[5] =", text[5])

# Отрицательные индексы идут с конца строки
print("text[-1] =", text[-1])
print("text[-2] =", text[-2])

# len() - длина строки
print("len(text) =", len(text))

print()

# Строки нельзя изменять по индексу (строка неизменяемая)
# Ошибка:
# text[0] = "p"

# lower() - нижний регистр
s1 = "Hello World"
print("lower():", s1.lower())

# upper() - верхний регистр
print("upper():", s1.upper())

# strip() - удаляет пробелы слева и справа
s2 = "   Python   "
print("before strip:", s2)
print("after strip:", s2.strip())

# Типичный случай с input() (для интерактива раскомментируй):
# user = input("Введите текст: ")
# print("normalized:", user.strip().lower())



# Срезы строк (slices)
# Срез позволяет получить часть строки.
# Формат: строка[start : end : step]

text = "Python"

# Получить подстроку с индекса start до end (end не включается)
print("text[0:3] =", text[0:3])   # 'Pyt'
print("text[2:5] =", text[2:5])   # 'tho'

# Если start не указан — берется с начала строки
print("text[:4] =", text[:4])     # 'Pyth'

# Если end не указан — берется до конца строки
print("text[3:] =", text[3:])     # 'hon'

# Срез всей строки
print("text[:] =", text[:])       # 'Python'

# Шаг (step)
print("text[::2] =", text[::2])   # 'Pto'
print("text[1::2] =", text[1::2]) # 'yhn'

# Отрицательный шаг — строка в обратном порядке
print("text[::-1] =", text[::-1]) # 'nohtyP'

print()

