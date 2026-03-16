import { Card } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";

const a = 3;

export function UserCard({ user, onPromote, onRemove }) {
  return (
    <Card
      title={user.name}
      footer={
        <div className="row">
          <span className="pill">{user.role}</span>
          <div className="row">
            <Button variant="ghost" onClick={() => onPromote(user.id)}>
              Повысить
            </Button>
            <Button variant="ghost" onClick={() => onRemove(user.id)}>
              Удалить
            </Button>

            {true && <h5>Hello</h5>}
            {false && <h5>World</h5>}

            {a==3 && <h5>Hello2</h5>}
          </div>
        </div>
      }
    >
      <p>
        Компонент <code>UserCard</code> ничего не знает о данных выше — он получает{" "}
        <code>user</code> и колбэки как пропсы.
      </p>
    </Card>
  );
}

