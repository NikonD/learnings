// 09_final_examples.js
// Тема 9: Итоговые примеры (условия, циклы, структуры данных)
// Запуск: node 09_final_examples.js

// На Windows включаем UTF-8 в консоли, чтобы русский текст не отображался кракозябрами
if (process.platform === "win32") {
  const { execSync } = require("child_process");
  try {
    execSync("chcp 65001", { stdio: "ignore" });
  } catch (_) {}
}

// ---------------------------------------------------------------------
// Пример 1: Угадай число
// Для интерактива: npm i readline-sync и раскомментировать вызов guessNumber() внизу.
// ---------------------------------------------------------------------

const func = (name) => {
  console.log(`Hello, ${name}`);
}

func("Nikon"); 
// -------------------------------

function guessNumber() {
  const readlineSync = require("readline-sync");
  const secret = Math.floor(Math.random() * 100) + 1;
  let tries = 0;
  while (true) {
    const guess = parseInt(readlineSync.question("Угадай число 1-100: "), 10);
    tries += 1;
    if (guess < secret) console.log("Мало");
    else if (guess > secret) console.log("Много");
    else {
      console.log(`Верно. Попыток: ${tries}`);
      break;
    }
  }
}

guessNumber();

// ---------------------------------------------------------------------
// Пример 2: Калькулятор (условия по оператору)
// ---------------------------------------------------------------------
function calculator(a, op, b) {
  if (op === "+") return a + b;
  if (op === "-") return a - b;
  if (op === "*") return a * b;
  if (op === "/") return b === 0 ? "Ошибка: деление на ноль" : a / b;
  if (op === "%") return b === 0 ? "Ошибка" : a % b;
  if (op === "**") return a ** b;
  return "Неизвестная операция";
}

console.log("calculator(10, '+', 5) =", calculator(10, "+", 5));
console.log("calculator(10, '/', 4) =", calculator(10, "/", 4));

// ---------------------------------------------------------------------
// Пример 3: Список покупок (массив + цикл)
// ---------------------------------------------------------------------
function shoppingList(itemsInput) {
  const items = itemsInput.filter((x) => x.trim() !== "");
  console.log("Список покупок:");
  items.forEach((x) => console.log("-", x));
}

shoppingList(["хлеб", "молоко", "яблоки"]);

// ---------------------------------------------------------------------
// Пример 4: Поиск в массиве (includes, indexOf)
// ---------------------------------------------------------------------
function findInList(arr, target) {
  const normalized = target.trim().toLowerCase();
  const found = arr.some((item) => item.toLowerCase() === normalized);
  return found ? "Найдено" : "Не найдено";
}

const fruits = ["apple", "banana", "orange"];
console.log("findInList(fruits, 'Banana') =", findInList(fruits, "Banana"));

// ---------------------------------------------------------------------
// Пример 5: Объект пользователя и перебор свойств
// ---------------------------------------------------------------------


// Раскомментируй для интерактивной игры (нужен readline-sync: npm i readline-sync):
