function Header({ page, onNavigate }) {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo">🔬</div>
        <div>
          <div className="header-title">MediScan Analyst</div>
          <div className="header-subtitle">AI Co-pilot for Medical Imaging</div>
        </div>
      </div>

      <nav className="header-nav">
        <button
          className={`nav-btn ${page === "landing" ? "active" : ""}`}
          onClick={() => onNavigate("landing")}
        >
          Home
        </button>
        <button
          className={`nav-btn ${page === "dashboard" ? "active" : ""}`}
          onClick={() => onNavigate("dashboard")}
        >
          Dashboard
        </button>
      </nav>

      <div className="header-status">
        <span className="status-dot" />
        System Online
      </div>
    </header>
  );
}

export default Header;
