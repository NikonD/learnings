# Модуль 5: JavaScript и TypeScript

Учебные материалы по основам JS и TS, по структуре аналогично модулю 2 (Python).

## Как запускать

### JavaScript (файлы `.js`)

Нужен [Node.js](https://nodejs.org/). В терминале из папки `module5`:

```bash
node 01_variables_and_types.js
node 02_console_and_io.js
# и т.д.
```

### TypeScript (файлы `.ts`)

Установить зависимости и запускать через `ts-node`:

```bash
npm install
npx ts-node 10_typescript_types.ts
npx ts-node 11_typescript_practice.ts
```

Или скомпилировать в JS и запустить:

```bash
npx tsc
node dist/10_typescript_types.js
```

## Содержание

| №  | Файл | Тема |
|----|------|------|
| 01 | 01_variables_and_types.js | Переменные и типы (number, string, boolean, const/let) |
| 02 | 02_console_and_io.js | Вывод в консоль, ввод (Node и браузер) |
| 03 | 03_builtin_and_operators.js | typeof, Math, операторы, длина, slice |
| 04 | 04_conditions.js | if / else if / else, ===, &&, \|\|, ! |
| 05 | 05_loops.js | for, while, for...of, for...in, break, continue |
| 06 | 06_string_formatting.js | Шаблонные литералы \`${}\`, конкатенация |
| 07 | 07_data_structures.js | Array, Object, Map, Set |
| 08 | 08_strings_basics.js | Индексы, slice, toLowerCase, trim, split, includes |
| 09 | 09_final_examples.js | Калькулятор, список, поиск, объекты |
| 10 | 10_typescript_types.ts | Типы, интерфейсы, union, literal types |
| 11 | 11_typescript_practice.ts | Типизированные функции, enum, generic, async |
