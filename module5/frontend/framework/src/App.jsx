import { useState, useEffect } from "react";
import { api, checkApi } from "./api";
import "./App.css";

const TABS = ["companies", "positions", "users"];

function App() {
  const [tab, setTab] = useState("companies");
  const [apiOk, setApiOk] = useState(null);
  const [companies, setCompanies] = useState([]);
  const [positions, setPositions] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkApi().then(setApiOk).catch(() => setApiOk(false));
  }, []);

  const loadCompanies = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.companies.list();
      setCompanies(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPositions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.positions.list();
      setPositions(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const loadUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.users.list();
      setUsers(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (tab === "companies") loadCompanies();
    if (tab === "positions") loadPositions();
    if (tab === "users") loadUsers();
  }, [tab]);

  return (
    <div className="app">
      <header>
        <h1>Company API</h1>
        <p className="subtitle">React-клиент для учебного бэкенда (module3/company_api_with_orm)</p>
        <p className={`api-status ${apiOk === true ? "ok" : apiOk === false ? "err" : ""}`}>
          {apiOk === null ? "Проверка API…" : apiOk ? "API доступен" : "API недоступен. Запустите бэкенд на :8000"}
        </p>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "companies" && "Компании"}
            {t === "positions" && "Должности"}
            {t === "users" && "Пользователи"}
          </button>
        ))}
      </nav>

      <main>
        {error && <p className="error">{error}</p>}
        {tab === "companies" && (
          <CompaniesSection
            list={companies}
            loading={loading}
            onReload={loadCompanies}
          />
        )}
        {tab === "positions" && (
          <PositionsSection
            list={positions}
            loading={loading}
            onReload={loadPositions}
          />
        )}
        {tab === "users" && (
          <UsersSection
            list={users}
            loading={loading}
            onReload={loadUsers}
          />
        )}
      </main>
    </div>
  );
}

function CompaniesSection({ list, loading, onReload }) {
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const openCreate = () => {
    setEditing(null);
    setName("");
    setDescription("");
    setFormOpen(true);
  };

  const openEdit = (c) => {
    setEditing(c);
    setName(c.name);
    setDescription(c.description || "");
    setFormOpen(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editing) {
        await api.companies.update(editing.id, { name: name.trim(), description: description.trim() || null });
      } else {
        await api.companies.create({ name: name.trim(), description: description.trim() || null });
      }
      setFormOpen(false);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm("Удалить компанию?")) return;
    try {
      await api.companies.delete(id);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <section className="panel">
      <h2>Компании</h2>
      <button type="button" className="btn primary" onClick={openCreate}>+ Новая компания</button>
      {loading && <p className="loading">Загрузка…</p>}
      {!loading && list.length === 0 && <p className="empty">Нет компаний</p>}
      {!loading && list.length > 0 && (
        <div className="list">
          {list.map((c) => (
            <div key={c.id} className="card">
              <div>
                <span className="title">{c.name}</span>
                {c.description && <div className="meta">{c.description}</div>}
              </div>
              <div className="card-actions">
                <button type="button" className="btn secondary" onClick={() => openEdit(c)}>Изменить</button>
                <button type="button" className="btn danger" onClick={() => handleDelete(c.id)}>Удалить</button>
              </div>
            </div>
          ))}
        </div>
      )}
      {formOpen && (
        <form className="form card" onSubmit={handleSubmit}>
          <h3>{editing ? "Редактировать компанию" : "Новая компания"}</h3>
          <label>Название <input value={name} onChange={(e) => setName(e.target.value)} required /></label>
          <label>Описание <input value={description} onChange={(e) => setDescription(e.target.value)} /></label>
          <div className="form-actions">
            <button type="submit" className="btn primary">Сохранить</button>
            <button type="button" className="btn secondary" onClick={() => setFormOpen(false)}>Отмена</button>
          </div>
        </form>
      )}
    </section>
  );
}

function PositionsSection({ list, loading, onReload }) {
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [title, setTitle] = useState("");
  const [companyId, setCompanyId] = useState("");

  const openCreate = () => {
    setEditing(null);
    setTitle("");
    setCompanyId("");
    setFormOpen(true);
  };

  const openEdit = (p) => {
    setEditing(p);
    setTitle(p.title);
    setCompanyId(String(p.company_id));
    setFormOpen(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editing) {
        await api.positions.update(editing.id, { title: title.trim(), company_id: Number(companyId) });
      } else {
        await api.positions.create({ title: title.trim(), company_id: Number(companyId) });
      }
      setFormOpen(false);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm("Удалить должность?")) return;
    try {
      await api.positions.delete(id);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <section className="panel">
      <h2>Должности</h2>
      <button type="button" className="btn primary" onClick={openCreate}>+ Новая должность</button>
      {loading && <p className="loading">Загрузка…</p>}
      {!loading && list.length === 0 && <p className="empty">Нет должностей</p>}
      {!loading && list.length > 0 && (
        <div className="list">
          {list.map((p) => (
            <div key={p.id} className="card">
              <div>
                <span className="title">{p.title}</span>
                <div className="meta">Компания: {p.company_name || "id " + p.company_id}</div>
              </div>
              <div className="card-actions">
                <button type="button" className="btn secondary" onClick={() => openEdit(p)}>Изменить</button>
                <button type="button" className="btn danger" onClick={() => handleDelete(p.id)}>Удалить</button>
              </div>
            </div>
          ))}
        </div>
      )}
      {formOpen && (
        <form className="form card" onSubmit={handleSubmit}>
          <h3>{editing ? "Редактировать должность" : "Новая должность"}</h3>
          <label>Название <input value={title} onChange={(e) => setTitle(e.target.value)} required /></label>
          <label>Компания (ID) <input type="number" value={companyId} onChange={(e) => setCompanyId(e.target.value)} required min={1} /></label>
          <div className="form-actions">
            <button type="submit" className="btn primary">Сохранить</button>
            <button type="button" className="btn secondary" onClick={() => setFormOpen(false)}>Отмена</button>
          </div>
        </form>
      )}
    </section>
  );
}

function UsersSection({ list, loading, onReload }) {
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [positionIdsStr, setPositionIdsStr] = useState("");

  const parsePositionIds = (s) =>
    s
      .split(/[\s,]+/)
      .map((x) => parseInt(x, 10))
      .filter((n) => !isNaN(n));

  const openCreate = () => {
    setEditing(null);
    setName("");
    setEmail("");
    setPositionIdsStr("");
    setFormOpen(true);
  };

  const openEdit = (u) => {
    setEditing(u);
    setName(u.name);
    setEmail(u.email);
    setPositionIdsStr((u.position_ids || []).join(", "));
    setFormOpen(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const position_ids = parsePositionIds(positionIdsStr);
    try {
      if (editing) {
        await api.users.update(editing.id, { name: name.trim(), email: email.trim(), position_ids });
      } else {
        await api.users.create({ name: name.trim(), email: email.trim(), position_ids });
      }
      setFormOpen(false);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm("Удалить пользователя?")) return;
    try {
      await api.users.delete(id);
      onReload();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <section className="panel">
      <h2>Пользователи</h2>
      <button type="button" className="btn primary" onClick={openCreate}>+ Новый пользователь</button>
      {loading && <p className="loading">Загрузка…</p>}
      {!loading && list.length === 0 && <p className="empty">Нет пользователей</p>}
      {!loading && list.length > 0 && (
        <div className="list">
          {list.map((u) => (
            <div key={u.id} className="card">
              <div>
                <span className="title">{u.name}</span>
                <div className="meta">{u.email} · должности: {(u.position_ids || []).join(", ") || "—"}</div>
              </div>
              <div className="card-actions">
                <button type="button" className="btn secondary" onClick={() => openEdit(u)}>Изменить</button>
                <button type="button" className="btn danger" onClick={() => handleDelete(u.id)}>Удалить</button>
              </div>
            </div>
          ))}
        </div>
      )}
      {formOpen && (
        <form className="form card" onSubmit={handleSubmit}>
          <h3>{editing ? "Редактировать пользователя" : "Новый пользователь"}</h3>
          <label>Имя <input value={name} onChange={(e) => setName(e.target.value)} required /></label>
          <label>Email <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
          <label>ID должностей через запятую <input value={positionIdsStr} onChange={(e) => setPositionIdsStr(e.target.value)} placeholder="1, 2" /></label>
          <div className="form-actions">
            <button type="submit" className="btn primary">Сохранить</button>
            <button type="button" className="btn secondary" onClick={() => setFormOpen(false)}>Отмена</button>
          </div>
        </form>
      )}
    </section>
  );
}

export default App;
