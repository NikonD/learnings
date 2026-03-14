// 10_typescript_types.ts
// Тема 10: TypeScript — типы, аннотации, интерфейсы
// Запуск: npx ts-node 10_typescript_types.ts  (или: tsc 10_typescript_types.ts && node 10_typescript_types.js)

// TypeScript — это JavaScript с указанием типов. Типы проверяются при компиляции.

// ---------------------------------------------------------------------
// Аннотации типов переменных
// ---------------------------------------------------------------------
let a: number = 10;
let name: string = "Alex";
let isActive: boolean = true;

// Массив чисел
let numbers: number[] = [1, 2, 3];
// или: Array<number>
let list: Array<number> = [1, 2, 3];

// Массив строк
let words: string[] = ["one", "two"];

// ---------------------------------------------------------------------
// Типы функций: аргументы и возвращаемое значение
// ---------------------------------------------------------------------
function add(x: number, y: number): number {
  return x + y;
}
console.log("add(2, 3) =", add(2, 3));

function greet(name: string): string {
  return `Hello, ${name}`;
}

// Функция без возвращаемого значения (void)
function logMessage(msg: string): void {
  console.log(msg);
}

// ---------------------------------------------------------------------
// Object и интерфейсы (описание формы объекта)
// ---------------------------------------------------------------------
interface User {
  name: string;
  age: number;
  city?: string; // необязательное поле
}

const user: User = {
  name: "Dana",
  age: 19,
  city: "Astana",
};

// city можно не указывать — оно опционально
const user2: User = { name: "Alex", age: 20 };

// ---------------------------------------------------------------------
// Union (объединение типов): значение может быть одним из типов
// ---------------------------------------------------------------------
let id: number | string;
id = 101;
id = "user-101";

function printId(id: number | string): void {
  if (typeof id === "string") {
    console.log("ID (string):", id);
  } else {
    console.log("ID (number):", id);
  }
}

// ---------------------------------------------------------------------
// Literal types — точные значения
// ---------------------------------------------------------------------
type Color = "red" | "green" | "blue";
let color: Color = "red";
// color = "yellow"; // ошибка компиляции

// ---------------------------------------------------------------------
// any — отключение проверки (стараться не использовать)
// ---------------------------------------------------------------------
let something: any = 42;
something = "hello";
something = true;

// ---------------------------------------------------------------------
// Тип по выводу: TypeScript сам выводит тип, если не указан
// ---------------------------------------------------------------------
let x = 5;       // x: number
let s = "text";  // s: string
let arr = [1, 2]; // arr: number[]