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

