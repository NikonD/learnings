# 09_final_examples.py
# Запуск: python 09_final_examples.py
#

# Программа 1: Угадай число (условия + циклы + import)
def guess_number():
    import random

    secret = random.randint(1, 100)
    tries = 0

    while True:
        guess = int(input("Угадай число 1-100: "))
        tries += 1

        if guess < secret:
            print("Мало")
        elif guess > secret:
            print("Много")
        else:
            print(f"Верно. Попыток: {tries}")
            break


# Программа 2: Калькулятор (ввод + условия)
# Операции: +, -, *, /, //, %, **
def calculator():
    a = float(input("A: "))
    op = input("Операция (+, -, *, /, //, %, **): ").strip()
    b = float(input("B: "))

    if op == "+":
        print(f"Результат: {a + b}")
    elif op == "-":
        print(f"Результат: {a - b}")
    elif op == "*":
        print(f"Результат: {a * b}")
    elif op == "/":
        if b == 0:
            print("Ошибка: деление на ноль")
        else:
            print(f"Результат: {a / b}")
    elif op == "//":
        if b == 0:
            print("Ошибка: деление на ноль")
        else:
            print(f"Результат: {a // b}")
    elif op == "%":
        if b == 0:
            print("Ошибка: деление на ноль")
        else:
            print(f"Результат: {a % b}")
    elif op == "**":
        print(f"Результат: {a ** b}")
    else:
        print("Ошибка: неизвестная операция")


# Программа 3: Список покупок (list + цикл)
# Вводим товары, пока не введут stop. Затем печатаем список.
def shopping_list():
    items = []

    while True:
        item = input("Товар (или stop): ").strip()
        if item.lower() == "stop":
            break
        if item == "":
            print("Пустая строка не добавлена")
            continue
        items.append(item)

    print("Список покупок:")
    for x in items:
        print("-", x)


# Программа 4: Поиск в списке (оператор in)
def find_in_list():
    items = ["apple", "banana", "orange"]
    target = input("Что ищем? ").strip().lower()

    if target in items:
        print("Найдено")
    else:
        print("Не найдено")


# Запуск (раскомментируй одну строку):
# guess_number()
# calculator()
# shopping_list()
# find_in_list()

