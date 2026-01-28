# 03_builtin_functions.py
# Тема 3: Встроенные функции Python
# Запуск: python 03_builtin_functions.py

# Встроенные функции - это готовые функции, которые уже есть в Python.
# Их не нужно подключать, они доступны сразу.

# type(x) - возвращает тип значения
x = 10
y = "10"
print("type(x) =", type(x))
print("type(y) =", type(y))


# len(x) - возвращает длину строки, списка и других коллекций
text = "Python"
numbers = [1, 2, 3, 4]
print("len(text) =", len(text))
print("len(numbers) =", len(numbers))


# int(x) - преобразует значение в целое число
print("int('15') =", int("15"))
print("int(7.9) =", int(7.9))  # дробная часть отбрасывается

# float(x) - преобразует значение в число с дробной частью
print("float('3.14') =", float("3.14"))
print("float(5) =", float(5))

# str(x) - преобразует значение в строку
value = 100
print("str(value) =", str(value))

# bool(x) - преобразует значение в логическое
print("bool(0) =", bool(0))
print("bool(1) =", bool(1))
print("bool('') =", bool(""))
print("bool('text') =", bool("text"))


# round(x, n) - округляет число
print("round(3.14159, 2) =", round(3.14159, 2))
print("round(2.5) =", round(2.5))


# min(), max() - минимальное и максимальное значение
nums = [5, 2, 9, 1]
print("min(nums) =", min(nums))
print("max(nums) =", max(nums))


# sum() - сумма элементов
print("sum(nums) =", sum(nums))


# sorted(x) - возвращает новый отсортированный список
print("sorted(nums) =", sorted(nums))
print("nums =", nums)  # исходный список не изменился


# range() - последовательность чисел (часто используется в циклах)
print("list(range(5)) =", list(range(5)))
print("list(range(2, 10, 2)) =", list(range(2, 10, 2)))

# Арифметические операторы
print(7 + 2)
print(7 - 2)
print(7 * 2)
print(7 / 2)    # обычное деление
print(7 // 2)   # целочисленное деление
print(7 % 2)    # остаток от деления
print(2 ** 3)   # степень
