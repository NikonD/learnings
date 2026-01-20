// js-basics-demo.js
// Интерактивная база JavaScript (вывод на страницу): переменные, условия, циклы, массивы, функции, объекты, JSON
//
// ВАЖНО: необходимо открыть консоль разработчика и смотреть console. (F12->console)

const outEl = document.querySelector("#out");
const codeEl = document.querySelector("#code");

const snippets = {
  vars: `// const и let + типы + шаблонная строка
const courseName = "JS basics";
let studentsCount = 12;
studentsCount = studentsCount + 1;

const pi = 3.14;
const isOnline = true;
const nothing = null;
let notSet;

const msg = \`Курс: \${courseName}, студентов: \${studentsCount}\`;
// выводим значения`,
  conditions: `// if / else if / else + тернарный оператор + switch
const score = 82;
if (score >= 90) {
  // ...
} else if (score >= 70) {
  // ...
} else {
  // ...
}

const access = score >= 70 ? "доступ разрешён" : "доступ запрещён";

const role = "student";
switch (role) {
  case "admin": break;
  case "mentor": break;
  case "student": break;
  default: break;
}`,
  loops: `// for / while / for...of
for (let i = 1; i <= 3; i++) {
  // ...
}

let x = 3;
while (x > 0) {
  x--;
}

const names = ["Аня", "Илья", "Даша"];
for (const name of names) {
  // ...
}`,
  arrays: `// массивы: индекс, push/pop, map/filter/reduce
const nums = [1, 2, 3];
nums.push(4);
const last = nums.pop();

const squares = nums.map((n) => n * n);
const even = squares.filter((n) => n % 2 === 0);
const sum = nums.reduce((acc, n) => acc + n, 0);`,
  functions: `// функции: declaration / expression / arrow
function add(a, b) {
  return a + b;
}

const multiply = function (a, b) {
  return a * b;
};

const toUpper = (s) => s.toUpperCase();`,
  json: `// объект + JSON.stringify / JSON.parse
const user = { id: 1, name: "Nikon", scores: [90, 85, 88] };
const jsonText = JSON.stringify(user, null, 2);
const parsed = JSON.parse(jsonText);`,
};

function line(text = "") {
  outEl.textContent += text + "\n";
  console.log(text);
}

function header(title) {
  line("");
  line("=== " + title + " ===");
}

function clearOut() {
  outEl.textContent = "";
}

function setCode(text) {
  if (!codeEl) return;
  codeEl.textContent = text || "";
}

// 1) Переменные
function demoVars() {
  header("1) Переменные");

  // const — значение переменной нельзя переassign
  const courseName = "JS basics";

  // let — можно менять
  let studentsCount = 12;
  studentsCount = studentsCount + 1;

  // Типы данных (самые частые)
  const pi = 3.14;               // number
  const isOnline = true;         // boolean
  const nothing = null;          // null (явно “ничего”)
  let notSet;                    // undefined (не присвоено)

  // Шаблонная строка (template literal)
  const msg = `Курс: ${courseName}, студентов: ${studentsCount}`;

  line(msg);
  line("pi = " + pi);
  line("isOnline = " + isOnline);
  line("nothing = " + nothing);
  line("notSet = " + notSet);

  console.log({ courseName, studentsCount, pi, isOnline, nothing, notSet });
}

// 2) Условия
function demoConditions() {
  header("2) Условия");

  const score = 82;
  if (score >= 90) {
    line("Оценка: 5");
  } else if (score >= 70) {
    line("Оценка: 4");
  } else {
    line("Оценка: 3 или ниже");
  }

  // Тернарный оператор: короткая запись if/else
  const access = score >= 70 ? "доступ разрешён" : "доступ запрещён";
  line("Тернарный: " + access);

  // switch: удобно, когда сравниваем одно значение с несколькими вариантами
  const role = "student";
  switch (role) {
    case "admin":
      line("Роль: админ");
      break;
    case "mentor":
      line("Роль: ментор");
      break;
    case "student":
      line("Роль: студент");
      break;
    default:
      line("Роль: неизвестно");
  }
}

// 3) Циклы
function demoLoops() {
  header("3) Циклы");

  // for: когда знаем количество повторений
  line("for:");
  for (let i = 1; i <= 3; i++) {
    line("  i = " + i);
  }

  // while: когда повторяем “пока условие истинно”
  line("while:");
  let x = 3;
  while (x > 0) {
    line("  x = " + x);
    x--;
  }

  // for...of: удобно пробегать массив
  line("for...of:");
  const names = ["Аня", "Илья", "Даша"];
  for (const name of names) {
    line("  привет, " + name);
  }
}

// 4) Массивы
function demoArrays() {
  header("4) Массивы");

  const nums = [1, 2, 3];
  line("nums = " + JSON.stringify(nums));
  line("nums[0] = " + nums[0]);

  // push / pop
  nums.push(4); // добавить в конец
  line("push(4) -> " + JSON.stringify(nums));
  const last = nums.pop(); // убрать с конца
  line("pop() -> last=" + last + ", nums=" + JSON.stringify(nums));

  // map: преобразовать каждый элемент → новый массив
  const squares = nums.map((n) => n * n);
  line("map(n*n) -> " + JSON.stringify(squares));

  // filter: оставить только те, которые подходят
  const even = squares.filter((n) => n % 2 === 0);
  line("filter(even) -> " + JSON.stringify(even));

  // reduce: “свернуть” массив в одно значение
  const sum = nums.reduce((acc, n) => acc + n, 0);
  line("reduce(sum) -> " + sum);
}

// 5) Функции
function demoFunctions() {
  header("5) Функции");

  // function declaration
  function add(a, b) {
    return a + b;
  }

  // function expression
  const multiply = function (a, b) {
    return a * b;
  };

  // arrow function
  const toUpper = (s) => s.toUpperCase();

  line("add(2, 3) = " + add(2, 3));
  line("multiply(2, 3) = " + multiply(2, 3));
  line('toUpper("hello") = ' + toUpper("hello"));
}

// 6) Объекты и JSON
function demoJson() {
  header("6) Объекты и JSON");

  // Объект: ключи и значения
  const user = {
    id: 1,
    name: "Nikon",
    isStudent: true,
    scores: [90, 85, 88],
  };

  // Доступ к полям
  line("user.name = " + user.name);
  line("user['isStudent'] = " + user["isStudent"]);

  // JSON.stringify — объект → строка
  const jsonText = JSON.stringify(user, null, 2);
  line("JSON.stringify(user):");
  line(jsonText);

  // JSON.parse — строка → объект
  const parsed = JSON.parse(jsonText);
  line("JSON.parse(...).scores[0] = " + parsed.scores[0]);

  console.log("user:", user);
  console.log("jsonText:", jsonText);
  console.log("parsed:", parsed);
}

// Кнопки управления (подключаем обработчики)
document.querySelector("#btn-clear").addEventListener("click", () => {
  setCode("");
  clearOut();
});
document.querySelector("#btn-vars").addEventListener("click", () => {
  setCode(snippets.vars);
  clearOut();
  demoVars();
});
document.querySelector("#btn-conditions").addEventListener("click", () => {
  setCode(snippets.conditions);
  clearOut();
  demoConditions();
});
document.querySelector("#btn-loops").addEventListener("click", () => {
  setCode(snippets.loops);
  clearOut();
  demoLoops();
});
document.querySelector("#btn-arrays").addEventListener("click", () => {
  setCode(snippets.arrays);
  clearOut();
  demoArrays();
});
document.querySelector("#btn-functions").addEventListener("click", () => {
  setCode(snippets.functions);
  clearOut();
  demoFunctions();
});
document.querySelector("#btn-json").addEventListener("click", () => {
  setCode(snippets.json);
  clearOut();
  demoJson();
});
document.querySelector("#btn-all").addEventListener("click", () => {
  setCode("// Запуск всех примеров подряд (смотри вывод справа)");
  clearOut();
  demoVars();
  demoConditions();
  demoLoops();
  demoArrays();
  demoFunctions();
  demoJson();
});

// Стартовое сообщение
clearOut();
setCode("// Нажми кнопку раздела — здесь появится код");
line("Готово. Нажми кнопку примера выше.");


