// 02_console_and_io.js
// Тема 2: Вывод в консоль и ввод данных
// Запуск: node 02_console_and_io.js

// ---------------------------------------------------------------------
// console.log() — вывод в консоль (аналог print в Python)
// ---------------------------------------------------------------------
console.log("Hello, world");
console.log(10 + 5);

let userName = "Alex";
let age = 20;
console.log("Name:", userName, "Age:", age);

// Несколько аргументов выводятся через пробел
console.log("A", "B", "C");

// ---------------------------------------------------------------------
// console с разными методами
// ---------------------------------------------------------------------
console.log("Обычный лог");
console.info("Информационное сообщение");
console.warn("Предупреждение");
console.error("Ошибка");

// ---------------------------------------------------------------------
// Ввод в Node.js (не в браузере)
// В браузере для ввода используют prompt() и alert().
// В Node.js нужен модуль readline или пакет readline-sync.
// ---------------------------------------------------------------------

// Простой ввод через readline (встроенный модуль Node.js):
const readline = require("readline");
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

// Асинхронный ввод — раскомментируй и запусти для проверки:
rl.question("Введите имя: ", (answer) => {
  console.log("Вы ввели:", answer);
  rl.close();
});

// Синхронный ввод проще делать через readline-sync: npm install readline-sync
const readlineSync = require('readline-sync');
const name = readlineSync.question('Введите имя: ');
const num = parseInt(readlineSync.question('Введите число: '), 10);

// ---------------------------------------------------------------------
// Кратко:
// - Вывод: console.log(), console.info(), console.warn(), console.error()
// - Ввод в браузере: prompt("Текст"), результат — строка
// - Ввод в Node: readline или readline-sync (см. выше)
// - Число из строки: Number(str) или parseInt(str, 10)