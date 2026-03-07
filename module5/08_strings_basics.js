// 08_strings_basics.js
// Тема 8: Строки — индексы, срезы, базовые методы
// Запуск: node 08_strings_basics.js

// Строка — последовательность символов. Индексы с 0.

let text = "JavaScript";

console.log("text =", text);
console.log("text[0] =", text[0]);
console.log("text[1] =", text[1]);
console.log("text[4] =", text[4]);

// Отрицательные индексы (только в новом JS или через slice)
console.log("text[text.length - 1] =", text[text.length - 1]);
console.log("text.slice(-1) =", text.slice(-1));
console.log("text.slice(-2) =", text.slice(-2));

// Длина
console.log("text.length =", text.length);

// Строки неизменяемы: text[0] = 'j' не изменит строку.

// ---------------------------------------------------------------------
// Методы: toLowerCase(), toUpperCase(), trim()
// ---------------------------------------------------------------------
let s1 = "Hello World";
console.log("toLowerCase():", s1.toLowerCase());
console.log("toUpperCase():", s1.toUpperCase());

let s2 = "   JavaScript   ";
console.log("before trim:", s2);
console.log("after trim:", s2.trim());

// ---------------------------------------------------------------------
// Срезы: slice(start, end) — от start до end (end не входит)
// ---------------------------------------------------------------------
text = "JavaScript";

console.log("text.slice(0, 4) =", text.slice(0, 4));   // 'Java'
console.log("text.slice(4, 10) =", text.slice(4, 10)); // 'Script'
console.log("text.slice(4) =", text.slice(4));         // до конца
console.log("text.slice(0, -6) =", text.slice(0, -6)); // 'Java'

// substring(start, end) — похоже на slice, но не поддерживает отрицательные индексы
console.log("text.substring(0, 4) =", text.substring(0, 4));

// ---------------------------------------------------------------------
// Поиск: includes(), indexOf(), startsWith(), endsWith()
// ---------------------------------------------------------------------
console.log("text.includes('Script') =", text.includes("Script"));
console.log("text.indexOf('a') =", text.indexOf("a"));
console.log("text.startsWith('Java') =", text.startsWith("Java"));
console.log("text.endsWith('Script') =", text.endsWith("Script"));

// ---------------------------------------------------------------------
// Разделение и склейка: split(), join()
// ---------------------------------------------------------------------
let words = "one two three";
console.log("split(' ') =", words.split(" "));
console.log("['a','b','c'].join('-') =", ["a", "b", "c"].join("-"));