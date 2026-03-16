import { useState } from "react";
import { Card } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";

function CounterValue({ value }) {
  return <div className="counter-value">{value}</div>;
}

function CounterControls({ onDec, onInc }) {
  return (
    <div className="row">
      <Button variant="ghost" onClick={onDec}>
        − 1
      </Button>
      <Button onClick={onInc}>+ 1</Button>
    </div>
  );
}

export function CounterCard() {
  const [value, setValue] = useState(0);

  return (
    <Card
      title="Мелкие переиспользуемые компоненты"
      footer={<span className="muted">Логика в одном месте, UI разбит на части.</span>}
    >
      <p>
        Вместо одного «толстого» компонента мы выделяем отдельные <code>CounterValue</code> и{" "}
        <code>CounterControls</code>, которые можно переиспользовать.
      </p>
      <CounterValue value={value} />
      <CounterControls
        onDec={() => setValue((v) => v - 1)}
        onInc={() => setValue((v) => v + 1)}
      />
    </Card>
  );
}

