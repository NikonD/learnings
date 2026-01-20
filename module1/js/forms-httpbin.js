// forms-httpbin.js
// Пример: перехват submit, сбор данных из формы и отправка на https://httpbin.org/post

const form = document.querySelector("#js-form");
const statusEl = document.querySelector("#status");
const outputEl = document.querySelector("#output");

function formDataToObject(formData) {
  const obj = {};
  for (const [key, value] of formData.entries()) {
    if (obj[key]) {
      obj[key] = Array.isArray(obj[key]) ? obj[key] : [obj[key]];
      obj[key].push(value);
    } else {
      obj[key] = value;
    }
  }
  return obj;
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    statusEl.textContent = "Отправляем...";
    outputEl.textContent = "{}";

    const formData = new FormData(form);

    // Можно посмотреть, что отправляем (для обучения)
    console.log("Отправляем:", formDataToObject(formData));

    try {
      const res = await fetch("https://httpbin.org/post", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("HTTP " + res.status);

      const json = await res.json();

      statusEl.textContent = "Готово. Смотри поле form:";
      outputEl.textContent = JSON.stringify(json.form, null, 2);
    } catch (err) {
      statusEl.textContent = "Ошибка: " + err.message;
      outputEl.textContent = "{}";
    }
  });
}


