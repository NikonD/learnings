import { Layout } from "./Layout.jsx";
import { UserListContainer } from "../users/UserListContainer.jsx";
import { CounterCard } from "../counter/CounterCard.jsx";

export function App() {
  return (
    <Layout
      title="React компоненты: контейнеры, презентационные и переиспользуемые части"
      subtitle="Учебный пример: как делить интерфейс на компоненты"
    >
      <div className="grid">
        <UserListContainer />
        <CounterCard />
      </div>
    </Layout>
  );
}

