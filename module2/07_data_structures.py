# 07_data_structures.py
# Тема 7: Структуры данных (list, dict)
# Запуск: python 07_data_structures.py

# Структуры данных используются для хранения нескольких значений.

# 1) list (список)
# Список - упорядоченная коллекция элементов.
numbers = [10, 20, 30]
names = ["Alex", "Bob", "Kate"]

print("numbers =", numbers)
print("names =", names)

# Доступ по индексу (нумерация начинается с 0)
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

# Перебор списка циклом for
for n in numbers:
    print("element:", n)

print()

# Удаление элемента
numbers.remove(25)
print("after remove:", numbers)

# Удаление последнего элемента
last = numbers.pop()
print("popped:", last)
print("after pop:", numbers)

# Проверка наличия элемента
print(30 in numbers)


# 2) dict (словарь)
# Словарь хранит пары: ключ -> значение.
student = {
    "name": "Dana",
    "age": 19,
    "city": "Astana"
}

print("student =", student)

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

