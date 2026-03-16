export function Card({ title, children, footer }) {
  return (
    <section className="card">
      {title && <h2>{title}</h2>}
      {children}
      {footer && <div className="card-footer">{footer}</div>}
    </section>
  );
}

