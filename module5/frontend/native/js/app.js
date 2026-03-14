(function () {
  const apiStatus = document.getElementById("apiStatus");

  function setApiStatus(ok, text) {
    apiStatus.textContent = text || (ok ? "API доступен" : "API недоступен");
    apiStatus.className = "api-status " + (ok ? "ok" : "err");
  }

  async function checkApi() {
    try {
      const r = await fetch(API_BASE + "/");
      setApiStatus(r.ok, r.ok ? "API доступен" : "Ошибка " + r.status);
    } catch (e) {
      setApiStatus(false, "API недоступен. Запустите бэкенд: module3/company_api_with_orm");
    }
  }

  // --- Вкладки ---
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const id = "section-" + tab.dataset.tab;
      document.getElementById(id).classList.add("active");
      if (tab.dataset.tab === "companies") loadCompanies();
      if (tab.dataset.tab === "positions") loadPositions();
      if (tab.dataset.tab === "users") loadUsers();
    });
  });

  // --- Компании ---
  const companiesList = document.getElementById("companiesList");
  const formCompany = document.getElementById("formCompany");
  const formCompanyTitle = document.getElementById("formCompanyTitle");
  const companyId = document.getElementById("companyId");
  const companyName = document.getElementById("companyName");
  const companyDescription = document.getElementById("companyDescription");

  async function loadCompanies() {
    companiesList.innerHTML = "<span class='loading'>Загрузка…</span>";
    try {
      const items = await api.companies.list();
      if (items.length === 0) {
        companiesList.innerHTML = "<span class='empty'>Нет компаний</span>";
        return;
      }
      companiesList.innerHTML = items
        .map(
          (c) =>
            `<div class="card" data-id="${c.id}">
              <div>
                <span class="title">${escapeHtml(c.name)}</span>
                ${c.description ? `<div class="meta">${escapeHtml(c.description)}</div>` : ""}
              </div>
              <div class="card-actions">
                <button type="button" class="btn secondary btnEditCompany" data-id="${c.id}">Изменить</button>
                <button type="button" class="btn danger btnDeleteCompany" data-id="${c.id}">Удалить</button>
              </div>
            </div>`
        )
        .join("");
      companiesList.querySelectorAll(".btnEditCompany").forEach((b) =>
        b.addEventListener("click", () => editCompany(Number(b.dataset.id)))
      );
      companiesList.querySelectorAll(".btnDeleteCompany").forEach((b) =>
        b.addEventListener("click", () => deleteCompany(Number(b.dataset.id)))
      );
    } catch (e) {
      companiesList.innerHTML = "<span class='error'>" + escapeHtml(e.message) + "</span>";
    }
  }

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  document.getElementById("btnNewCompany").addEventListener("click", () => {
    formCompanyTitle.textContent = "Новая компания";
    companyId.value = "";
    companyName.value = "";
    companyDescription.value = "";
    formCompany.classList.remove("hidden");
  });

  document.querySelector(".btnCancelCompany").addEventListener("click", () => formCompany.classList.add("hidden"));

  async function editCompany(id) {
    const c = await api.companies.get(id);
    formCompanyTitle.textContent = "Редактировать компанию";
    companyId.value = c.id;
    companyName.value = c.name;
    companyDescription.value = c.description || "";
    formCompany.classList.remove("hidden");
  }

  formCompany.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = companyId.value ? Number(companyId.value) : null;
    const data = { name: companyName.value.trim(), description: companyDescription.value.trim() || null };
    try {
      if (id) await api.companies.update(id, data);
      else await api.companies.create(data);
      formCompany.classList.add("hidden");
      loadCompanies();
    } catch (err) {
      alert(err.message);
    }
  });

  async function deleteCompany(id) {
    if (!confirm("Удалить компанию?")) return;
    try {
      await api.companies.delete(id);
      loadCompanies();
    } catch (err) {
      alert(err.message);
    }
  }

  // --- Должности ---
  const positionsList = document.getElementById("positionsList");
  const formPosition = document.getElementById("formPosition");
  const formPositionTitle = document.getElementById("formPositionTitle");
  const positionId = document.getElementById("positionId");
  const positionTitle = document.getElementById("positionTitle");
  const positionCompanyId = document.getElementById("positionCompanyId");

  async function loadPositions() {
    positionsList.innerHTML = "<span class='loading'>Загрузка…</span>";
    try {
      const items = await api.positions.list();
      if (items.length === 0) {
        positionsList.innerHTML = "<span class='empty'>Нет должностей</span>";
        return;
      }
      positionsList.innerHTML = items
        .map(
          (p) =>
            `<div class="card" data-id="${p.id}">
              <div>
                <span class="title">${escapeHtml(p.title)}</span>
                <div class="meta">Компания: ${escapeHtml(p.company_name || "id " + p.company_id)}</div>
              </div>
              <div class="card-actions">
                <button type="button" class="btn secondary btnEditPosition" data-id="${p.id}">Изменить</button>
                <button type="button" class="btn danger btnDeletePosition" data-id="${p.id}">Удалить</button>
              </div>
            </div>`
        )
        .join("");
      positionsList.querySelectorAll(".btnEditPosition").forEach((b) =>
        b.addEventListener("click", () => editPosition(Number(b.dataset.id)))
      );
      positionsList.querySelectorAll(".btnDeletePosition").forEach((b) =>
        b.addEventListener("click", () => deletePosition(Number(b.dataset.id)))
      );
    } catch (e) {
      positionsList.innerHTML = "<span class='error'>" + escapeHtml(e.message) + "</span>";
    }
  }

  document.getElementById("btnNewPosition").addEventListener("click", () => {
    formPositionTitle.textContent = "Новая должность";
    positionId.value = "";
    positionTitle.value = "";
    positionCompanyId.value = "";
    formPosition.classList.remove("hidden");
  });

  document.querySelector(".btnCancelPosition").addEventListener("click", () => formPosition.classList.add("hidden"));

  async function editPosition(id) {
    const p = await api.positions.get(id);
    formPositionTitle.textContent = "Редактировать должность";
    positionId.value = p.id;
    positionTitle.value = p.title;
    positionCompanyId.value = p.company_id;
    formPosition.classList.remove("hidden");
  }

  formPosition.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = positionId.value ? Number(positionId.value) : null;
    const data = { title: positionTitle.value.trim(), company_id: Number(positionCompanyId.value) };
    try {
      if (id) await api.positions.update(id, data);
      else await api.positions.create(data);
      formPosition.classList.add("hidden");
      loadPositions();
    } catch (err) {
      alert(err.message);
    }
  });

  async function deletePosition(id) {
    if (!confirm("Удалить должность?")) return;
    try {
      await api.positions.delete(id);
      loadPositions();
    } catch (err) {
      alert(err.message);
    }
  }

  // --- Пользователи ---
  const usersList = document.getElementById("usersList");
  const formUser = document.getElementById("formUser");
  const formUserTitle = document.getElementById("formUserTitle");
  const userId = document.getElementById("userId");
  const userName = document.getElementById("userName");
  const userEmail = document.getElementById("userEmail");
  const userPositionIds = document.getElementById("userPositionIds");

  async function loadUsers() {
    usersList.innerHTML = "<span class='loading'>Загрузка…</span>";
    try {
      const items = await api.users.list();
      if (items.length === 0) {
        usersList.innerHTML = "<span class='empty'>Нет пользователей</span>";
        return;
      }
      usersList.innerHTML = items
        .map(
          (u) =>
            `<div class="card" data-id="${u.id}">
              <div>
                <span class="title">${escapeHtml(u.name)}</span>
                <div class="meta">${escapeHtml(u.email)} · должности: ${(u.position_ids || []).join(", ") || "—"}</div>
              </div>
              <div class="card-actions">
                <button type="button" class="btn secondary btnEditUser" data-id="${u.id}">Изменить</button>
                <button type="button" class="btn danger btnDeleteUser" data-id="${u.id}">Удалить</button>
              </div>
            </div>`
        )
        .join("");
      usersList.querySelectorAll(".btnEditUser").forEach((b) =>
        b.addEventListener("click", () => editUser(Number(b.dataset.id)))
      );
      usersList.querySelectorAll(".btnDeleteUser").forEach((b) =>
        b.addEventListener("click", () => deleteUser(Number(b.dataset.id)))
      );
    } catch (e) {
      usersList.innerHTML = "<span class='error'>" + escapeHtml(e.message) + "</span>";
    }
  }

  document.getElementById("btnNewUser").addEventListener("click", () => {
    formUserTitle.textContent = "Новый пользователь";
    userId.value = "";
    userName.value = "";
    userEmail.value = "";
    userPositionIds.value = "";
    formUser.classList.remove("hidden");
  });

  document.querySelector(".btnCancelUser").addEventListener("click", () => formUser.classList.add("hidden"));

  async function editUser(id) {
    const u = await api.users.get(id);
    formUserTitle.textContent = "Редактировать пользователя";
    userId.value = u.id;
    userName.value = u.name;
    userEmail.value = u.email;
    userPositionIds.value = (u.position_ids || []).join(", ");
    formUser.classList.remove("hidden");
  }

  function parsePositionIds(s) {
    return s
      .split(/[\s,]+/)
      .map((x) => parseInt(x, 10))
      .filter((n) => !isNaN(n));
  }

  formUser.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = userId.value ? Number(userId.value) : null;
    const data = {
      name: userName.value.trim(),
      email: userEmail.value.trim(),
      position_ids: parsePositionIds(userPositionIds.value),
    };
    try {
      if (id) await api.users.update(id, data);
      else await api.users.create(data);
      formUser.classList.add("hidden");
      loadUsers();
    } catch (err) {
      alert(err.message);
    }
  });

  async function deleteUser(id) {
    if (!confirm("Удалить пользователя?")) return;
    try {
      await api.users.delete(id);
      loadUsers();
    } catch (err) {
      alert(err.message);
    }
  }

  // Инициализация
  checkApi();
  loadCompanies();
})();
