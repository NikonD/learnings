# 07_data_structures.py
# Тема 7: Структуры данных (list, tuple, dict, set)
# Запуск: python 07_data_structures.py

# Структуры данных используются для хранения нескольких значений в одной переменной.

# --------------------------------------------------
# 1) list (список)
# --------------------------------------------------
# list — упорядоченная коллекция элементов, которую МОЖНО изменять

numbers = [10, 20, 30]
names = ["Alex", "Bob", "Kate"]

print("numbers =", numbers)
print("names =", names)

# Создание списка через list()
numbers2 = list([1, 2, 3])
empty_list = list()

print("numbers2 =", numbers2)
print("empty_list =", empty_list)

# Доступ по индексу (индексация начинается с 0)
print("numbers[0] =", numbers[0])
print("names[1] =", names[1])

# Изменение элемента
numbers[1] = 25
print("numbers after change =", numbers)

# Добавление элемента
numbers.append(40)
print("numbers after append =", numbers)

# Длина списка
print("len(numbers) =", len(numbers))

# Перебор списка
for n in numbers:
    print("element:", n)

# Удаление элемента по значению
numbers.remove(25)
print("after remove:", numbers)

# Удаление последнего элемента
last = numbers.pop()
print("popped:", last)
print("after pop:", numbers)

# Проверка наличия элемента
print(30 in numbers)

print()


# --------------------------------------------------
# 2) tuple (кортеж)
# --------------------------------------------------
# tuple — упорядоченная коллекция, которую НЕЛЬЗЯ изменять

coords = (10, 20)
colors = tuple(["red", "green", "blue"])

print("coords =", coords)
print("colors =", colors)

# Создание пустого кортежа
empty_tuple = tuple()
print("empty_tuple =", empty_tuple)

# Доступ по индексу
print("coords[0] =", coords[0])

# Попытка изменения приведет к ошибке
# coords[0] = 5  # ошибка

print()


# --------------------------------------------------
# 3) dict (словарь)
# --------------------------------------------------
# dict — хранит данные в формате: ключ -> значение

student = {
    "name": "Dana",
    "age": 19,
    "city": "Astana"
}

print("student =", student)

# Создание словаря через dict()
student2 = dict(name="Alex", age=21)
empty_dict = dict()

print("student2 =", student2)
print("empty_dict =", empty_dict)

# Доступ по ключу
print("name =", student["name"])
print("age =", student["age"])

# Изменение значения
student["age"] = 20

# Добавление новой пары
student["group"] = "IS-101"

print("student after changes =", student)

# Перебор словаря
for key in student:
    print(key, "->", student[key])

print()


# --------------------------------------------------
# 4) set (множество)
# --------------------------------------------------
# set — хранит ТОЛЬКО уникальные значения
# Порядок элементов не гарантирован

numbers_set = {1, 2, 3, 3, 2}
letters = set(["a", "b", "a", "c"])
empty_set = set()

print("numbers_set =", numbers_set)
print("letters =", letters)
print("empty_set =", empty_set)

# Добавление элемента
numbers_set.add(4)
print("after add:", numbers_set)

# Проверка наличия элемента
print(2 in numbers_set)

print()


# --------------------------------------------------
# Краткое сравнение
# --------------------------------------------------
# list  — упорядочен, можно изменять, есть индексы
# tuple — упорядочен, нельзя изменять
# dict  — хранит пары ключ: значение
# set   — хранит только уникальные значения, без порядка
