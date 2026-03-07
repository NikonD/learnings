// 03_builtin_and_operators.js
// Тема 3: Встроенные функции и операторы JavaScript
// Запуск: node 03_builtin_and_operators.js

// ---------------------------------------------------------------------
// typeof x — возвращает строку с типом значения
// ---------------------------------------------------------------------
let x = 10;
let y = "10";
console.log("typeof x =", typeof x);
console.log("typeof y =", typeof y);
console.log("typeof {} =", typeof {});
console.log("typeof [] =", typeof []); // "object" (массив — тоже object!)

// ---------------------------------------------------------------------
// length — у строк и массивов (свойство, не функция)
// ---------------------------------------------------------------------
let text = "Python";
let numbers = [1, 2, 3, 4];
console.log("text.length =", text.length);
console.log("numbers.length =", numbers.length);

// ---------------------------------------------------------------------
// Number(), String(), Boolean() — преобразование типов
// ---------------------------------------------------------------------
console.log("Number('15') =", Number("15"));
console.log("Number('15.7') =", Number("15.7"));
console.log("parseInt('15.7', 10) =", parseInt("15.7", 10));
console.log("String(100) =", String(100));
console.log("Boolean(0) =", Boolean(0));
console.log("Boolean('') =", Boolean(""));
console.log("Boolean('text') =", Boolean("text"));

// ---------------------------------------------------------------------
// Math — встроенный объект для математики
// ---------------------------------------------------------------------
console.log("Math.round(3.14159) =", Math.round(3.14159));
console.log("Math.floor(3.7) =", Math.floor(3.7));
console.log("Math.ceil(3.2) =", Math.ceil(3.2));
console.log("Math.min(5, 2, 9, 1) =", Math.min(5, 2, 9, 1));
console.log("Math.max(5, 2, 9, 1) =", Math.max(5, 2, 9, 1));

let nums = [5, 2, 9, 1];
console.log("Math.min(...nums) =", Math.min(...nums));
console.log("Math.max(...nums) =", Math.max(...nums));

// ---------------------------------------------------------------------
// Массив: сортировка и сумма
// ---------------------------------------------------------------------
console.log("nums.sort() =", [...nums].sort());        // копия, чтобы не менять nums
console.log("nums =", nums);
console.log("nums.reduce((a,b)=>a+b, 0) =", nums.reduce((a, b) => a + b, 0));
// nums.reduce(function(a, b) { return a + b; }, 0);
// ---------------------------------------------------------------------
// Арифметические операторы
// ---------------------------------------------------------------------
console.log("7 + 2 =", 7 + 2);
console.log("7 - 2 =", 7 - 2);
console.log("7 * 2 =", 7 * 2);
console.log("7 / 2 =", 7 / 2);    // в JS всегда дробное деление
console.log("Math.floor(7/2) =", Math.floor(7 / 2));   // целочисленное «вручную»
console.log("7 % 2 =", 7 % 2);    // остаток
console.log("2 ** 3 =", 2 ** 3);  // степень

// ---------------------------------------------------------------------
// Строки: индексы и slice
// ---------------------------------------------------------------------
let s = "JavaScript";
console.log("s[0] =", s[0]);
console.log("s.slice(0, 4) =", s.slice(0, 4));
console.log("s.slice(4) =", s.slice(4));