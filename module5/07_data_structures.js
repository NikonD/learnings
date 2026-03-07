// 07_data_structures.js
// Тема 7: Структуры данных (массив, объект, Map, Set)
// Запуск: node 07_data_structures.js

// ---------------------------------------------------------------------
// 1) Array (массив) — упорядоченный список, можно изменять
// ---------------------------------------------------------------------
let numbers = [10, 20, 30];
let names = ["Alex", "Bob", "Kate"];

console.log("numbers =", numbers);
console.log("names =", names);

// Доступ по индексу (с 0)
console.log("numbers[0] =", numbers[0]);
console.log("names[1] =", names[1]);

// Изменение элемента
numbers[1] = 25;
console.log("numbers after change =", numbers);

// Добавление в конец
numbers.push(40);
console.log("numbers after push =", numbers);

// Длина
console.log("numbers.length =", numbers.length);

// Перебор
for (const n of numbers) {
  console.log("element:", n);
}

// Удаление: pop() — последний, splice(i, 1) — по индексу
let last = numbers.pop();
console.log("popped:", last, "| numbers =", numbers);

// Проверка наличия
console.log("30 in array:", numbers.includes(30));

// ---------------------------------------------------------------------
// 2) Object (объект) — пары ключ: значение (как dict в Python)
// ---------------------------------------------------------------------
let student = {
  name: "Dana",
  age: 19,
  city: "Astana",
};

console.log("student =", student);

// Доступ: через точку или квадратные скобки
console.log("student.name =", student.name);
console.log("student['age'] =", student["age"]);

// Изменение и добавление
student.age = 20;
student.group = "IS-101";
console.log("student after changes =", student);

// Перебор ключей
for (const key in student) {
  console.log(key, "->", student[key]);
}

// Вложенный объект
let user = {
  name: "Alex",
  address: { city: "Almaty", street: "Abay" },
};
console.log("user.address.city =", user.address.city);

// ---------------------------------------------------------------------
// 3) Map — коллекция ключ–значение (ключи могут быть любого типа)
// ---------------------------------------------------------------------
let map = new Map();
map.set("name", "Alex");
map.set(1, "one");
map.set(true, "yes");

console.log("map.get('name') =", map.get("name"));
console.log("map.has(1) =", map.has(1));
console.log("map.size =", map.size);

for (const [key, value] of map) {
  console.log(key, "->", value);
}

// ---------------------------------------------------------------------
// 4) Set — только уникальные значения, без дубликатов
// ---------------------------------------------------------------------
let numbersSet = new Set([1, 2, 3, 3, 2]);
console.log("numbersSet =", numbersSet);
numbersSet.add(4);
console.log("after add 4:", numbersSet);
console.log("numbersSet.has(2) =", numbersSet.has(2));

// ---------------------------------------------------------------------
// Кратко:
// Array — упорядоченный список, индексы с 0
// Object — пары ключ: значение, ключи — строки (или символы)
// Map    — ключ–значение, любые типы ключей
// Set    — только уникальные значения