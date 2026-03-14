// 11_typescript_practice.ts
// Тема 11: TypeScript на практике — типизированные функции и структуры
// Запуск: npx ts-node 11_typescript_practice.ts

// ---------------------------------------------------------------------
// Типизированный калькулятор
// ---------------------------------------------------------------------
type Operation = "+" | "-" | "*" | "/" | "%" | "**";

function calculator(a: number, op: Operation, b: number): number | string {
  if (op === "+") return a + b;
  if (op === "-") return a - b;
  if (op === "*") return a * b;
  if (op === "/") return b === 0 ? "Ошибка: деление на ноль" : a / b;
  if (op === "%") return b === 0 ? "Ошибка" : a % b;
  if (op === "**") return a ** b;
  return "Неизвестная операция";
}

console.log("calculator(10, '+', 5) =", calculator(10, "+", 5));

// ---------------------------------------------------------------------
// Интерфейс и массив объектов
// ---------------------------------------------------------------------
interface Product {
  id: number;
  name: string;
  price: number;
}

const products: Product[] = [
  { id: 1, name: "Хлеб", price: 150 },
  { id: 2, name: "Молоко", price: 320 },
  { id: 3, name: "Яблоки", price: 500 },
];

function totalPrice(items: Product[]): number {
  return items.reduce((sum, p) => sum + p.price, 0);
}

console.log("totalPrice(products) =", totalPrice(products));

// ---------------------------------------------------------------------
// Generic: функция, работающая с разными типами
// ---------------------------------------------------------------------
function firstElement<T>(arr: T[]): T | undefined {
  return arr[0];
}

function secondElement(arr: any[]): any {
  return arr[1];
}

console.log("firstElement([1,2,3]) =", firstElement([1, 2, 3]));
console.log("firstElement(['a','b']) =", firstElement(["a", "b"]));

// ---------------------------------------------------------------------
// Enum — перечисление (удобно для набора констант)
// ---------------------------------------------------------------------
enum Status {
  Pending,
  Done,
  Failed,
}

let status: Status = Status.Pending;
console.log("Status.Pending =", Status.Pending, "| status =", status);

// Строковый enum
enum Direction {
  Up = "UP",
  Down = "DOWN",
}
console.log("Direction.Up =", Direction.Up);

// ---------------------------------------------------------------------
// Тип возврата Promise (асинхронная функция)
// ---------------------------------------------------------------------
async function fetchUserName(id: number): Promise<string> {
  // заглушка вместо реального запроса
  return `user-${id}`;
}

fetchUserName(1).then((name) => console.log("fetchUserName(1) =", name));