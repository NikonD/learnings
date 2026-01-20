// js-showcase.js — базовые приёмы JavaScript в браузере
// Открывать вместе с: ../html/js-showcase.html

// 1) Получаем элементы со страницы (DOM)
const titleEl = document.querySelector("#title");
const logEl = document.querySelector("#log");

// 2) Функция для вывода строк в “лог” на странице
function log(message) {
  const line = document.createElement("div");
  line.textContent = message;
  logEl.appendChild(line);
}

// 3) Изменяем текст на странице
titleEl.textContent = "JS — витрина приёмов (заголовок изменён через JS)";

// 4) События: click
document.querySelector("#btn").addEventListener("click", () => {
  log("Клик по кнопке в " + new Date().toLocaleTimeString());
});

// 5) classList: переключаем класс на body
document.querySelector("#toggle").addEventListener("click", () => {
  document.body.classList.toggle("is-dark");
  log("Переключили класс is-dark на body");
});

// 6) dataset: читаем data-атрибут
const box = document.querySelector("#box");
log("data-id у #box = " + box.dataset.id);


