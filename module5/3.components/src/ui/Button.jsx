export function Button({ variant = "primary", className = "", children, ...props }) {
  const classes = ["btn", variant === "primary" ? "btn--primary" : "btn--ghost", className]
    .filter(Boolean)
    .join(" ");

  return (
    <button type="button" {...props} className={classes}>
      {children}
    </button>
  );
}

