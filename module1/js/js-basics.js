/*
  JS БАЗА (очень простая шпаргалка)
  Цель: прочитать и понять основы:
  - переменные
  - условия
  - циклы
  - массивы
  - функции
  - объекты + JSON

  Как запустить:
  - Открой файл module1/html/js-basics-cheatsheet.html
  - Открой DevTools (F12) → Console
*/

console.log("=== JS БАЗА: старт ===");

// ============================================================================
// 1) ПЕРЕМЕННЫЕ
// ============================================================================

// const — нельзя перезаписать переменную
const name = "Иван";
// name = "Петя"; // ❌ ошибка

// let — можно менять
let age = 20;
age = age + 1;

// Типы (самые частые)
const n = 10;          // number
const s = "строка";    // string
const b = true;        // boolean
const u = undefined;   // undefined (не задано)
const nul = null;      // null (явно “ничего”)

console.log("name:", name);   // Иван
console.log("age:", age);     // 21
// выведет типы переменных. пример: number, string, boolean, undefined, null
console.log("types:", typeof n, typeof s, typeof b, typeof u, nul);

// Шаблонные строки (удобно)
console.log(`Привет, ${name}. Тебе ${age} лет.`); // выведет: Привет, Иван. Тебе 20 лет.

// ============================================================================
// 2) УСЛОВИЯ (if/else) + тернарный оператор + switch
// ============================================================================

const score = 82;

if (score >= 90) {
  console.log("Оценка: 5");
} else if (score >= 70) {
  console.log("Оценка: 4"); // сюда попадём при score=82
} else {
  console.log("Оценка: 3 или ниже");
}

// Тернарный оператор: условие ? если_да : если_нет
const access = score >= 70 ? "доступ есть" : "доступа нет"; // при 82 -> доступ есть
console.log("access:", access); // "доступ есть"

// switch — удобно для нескольких вариантов одного значения
const role = "student";
switch (role) {
  case "admin":
    console.log("роль: админ");
    break;
  case "mentor":
    console.log("роль: ментор");
    break;
  case "student":
    console.log("роль: студент");
    break;
  default:
    console.log("роль: неизвестно");
}

// ============================================================================
// 3) ЦИКЛЫ (for / while / for...of)
// ============================================================================

console.log("for:");            // цикл 3 раза
for (let i = 1; i <= 3; i++) {
  console.log(" i =", i);
}

console.log("while:");
let x = 3;
while (x > 0) {
  console.log(" x =", x);
  x--;
}

console.log("for...of (по массиву):");
const names = ["Аня", "Илья", "Даша"];
for (const n1 of names) {
  console.log(" привет,", n1);
}

// ============================================================================
// 4) МАССИВЫ
// ============================================================================

const nums = [1, 2, 3];
console.log("nums:", nums);           // [1,2,3]
console.log("nums[0]:", nums[0]);     // индексы с 0 -> 1

nums.push(4);                         // добавить в конец
console.log("push ->", nums);         // [1,2,3,4]

const last = nums.pop();              // убрать последний
console.log("pop -> last:", last, "nums:", nums); // last=4, nums=[1,2,3]

// Полезные методы:
const squares = nums.map((v) => v * v);          // преобразовать каждый элемент
const even = squares.filter((v) => v % 2 === 0); // оставить подходящие
const sum = nums.reduce((acc, v) => acc + v, 0); // “свернуть” в одно число

console.log("map squares:", squares); // [1,4,9]
console.log("filter even:", even);    // [4]
console.log("reduce sum:", sum);      // 6

// ============================================================================
// 5) ФУНКЦИИ
// ============================================================================

// function declaration
function add(a, bb) {
  return a + bb;
}

// arrow function
const multiply = (a, bb) => a * bb;

console.log("add(2,3) =", add(2, 3));          // 5
console.log("multiply(2,3) =", multiply(2, 3)); // 6

// ============================================================================
// 6) ОБЪЕКТЫ и JSON
// ============================================================================

// Объект = ключ: значение
const user = {
  id: 1,
  name: "Nikon",
  isStudent: true,
  scores: [90, 85, 88],
};

console.log("user.name:", user.name);
console.log("user['isStudent']:", user["isStudent"]);

// JSON — это строка (текст) с данными
const jsonText = JSON.stringify(user);     // объект -> JSON-строка
const parsed = JSON.parse(jsonText);       // JSON-строка -> объект

console.log("JSON.stringify(user) ->", jsonText); // строка JSON
console.log("JSON.parse(...) ->", parsed);        // обратно объект

console.log("=== JS БАЗА: конец ===");


