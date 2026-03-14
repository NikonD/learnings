/**
 * Клиент API бэкенда (module3/company_api_with_orm).
 * Бэкенд должен быть запущен на http://localhost:8000
 */
const API_BASE = "http://localhost:8000";

async function request(path, options = {}) {
  const url = API_BASE + path;
  const config = {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  };
  if (config.body && typeof config.body === "object" && !(config.body instanceof FormData)) {
    config.body = JSON.stringify(config.body);
  }
  const res = await fetch(url, config);
  if (res.status === 204) return null;
  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try {
      const j = JSON.parse(text);
      detail = j.detail || text;
    } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return text ? JSON.parse(text) : null;
}

export const api = {
  companies: {
    list: () => request("/companies"),
    get: (id) => request(`/companies/${id}`),
    create: (data) => request("/companies", { method: "POST", body: data }),
    update: (id, data) => request(`/companies/${id}`, { method: "PUT", body: data }),
    delete: (id) => request(`/companies/${id}`, { method: "DELETE" }),
  },
  positions: {
    list: () => request("/positions/with-company"),
    get: (id) => request(`/positions/${id}`),
    create: (data) => request("/positions", { method: "POST", body: data }),
    update: (id, data) => request(`/positions/${id}`, { method: "PUT", body: data }),
    delete: (id) => request(`/positions/${id}`, { method: "DELETE" }),
  },
  users: {
    list: () => request("/users"),
    get: (id) => request(`/users/${id}`),
    create: (data) => request("/users", { method: "POST", body: data }),
    update: (id, data) => request(`/users/${id}`, { method: "PUT", body: data }),
    delete: (id) => request(`/users/${id}`, { method: "DELETE" }),
  },
};

export async function checkApi() {
  const r = await fetch(API_BASE + "/");
  return r.ok;
}
