import { useState } from "react";
import { Card } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
import { UserCard } from "./UserCard.jsx";

const INITIAL_USERS = [
  { id: 1, name: "Алиса", role: "Frontend" },
  { id: 2, name: "Борис", role: "Backend" },
];


export function UserListContainer() {
  const [users, setUsers] = useState(INITIAL_USERS);

  const promote = (id) =>
    setUsers((prev) =>
      prev.map((u) => (u.id === id ? { ...u, role: `${u.role} · Senior` } : u)),
    );

  const removeUser = (id) => setUsers((prev) => prev.filter((u) => u.id !== id));

  const reset = () => setUsers(INITIAL_USERS);

  return (
    <Card
      title="Контейнер + презентационные компоненты"
      footer={
        <div className="row">
          <span className="muted">Состояние хранится только в контейнере.</span>
          <Button variant="ghost" onClick={reset}>
            Сбросить список
          </Button>
        </div>
      }
    >
      <p>
        <strong>UserListContainer</strong> знает про массив пользователей и функции изменения, а{" "}
        <strong>UserCard</strong> занимается только отображением.
      </p>
      <div className="grid grid--cards">
        {users.map((user) => (
          <UserCard
            key={user.id}
            user={user}
            onPromote={promote}
            onRemove={removeUser}
          />
        ))}
      </div>
    </Card>
  );
}

