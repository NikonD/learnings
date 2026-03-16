export function Layout({ title, subtitle, children }) {
  return (
    <div className="page">
      <header className="page-header">
        <h1>{title}</h1>
        {subtitle && <p className="subtitle">{subtitle}</p>}
      </header>
      <main>{children}</main>
    </div>
  );
}

