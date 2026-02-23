function LandingPage({ onGetStarted }) {
  return (
    <main className="landing">
      <div className="landing-hero">
        <div className="landing-badge">⚡ Multi-Agent AI System</div>

        <h1>
          <span className="gradient">MediScan</span> Analyst
        </h1>

        <p className="landing-desc">
          An Agentic AI Co-pilot for Medical Imaging and Diagnosis Support.
          Three specialized agents work together to analyze X-rays, MRIs, and CT
          scans — delivering transparent, explainable diagnostic insights with
          human-in-the-loop oversight.
        </p>

        <div className="landing-actions">
          <button className="btn-primary" onClick={onGetStarted}>
            Start Analysis →
          </button>
          <button className="btn-secondary" onClick={onGetStarted}>
            View Dashboard
          </button>
        </div>
      </div>

      <div className="agents-showcase">
        <div className="agent-card">
          <div className="agent-card-icon vision">👁️</div>
          <h3>Vision Agent</h3>
          <p>
            Processes medical images to identify anatomical structures, detect
            abnormalities, and extract key features using advanced computer
            vision.
          </p>
        </div>

        <div className="agent-card">
          <div className="agent-card-icon analysis">🧠</div>
          <h3>Analysis Agent</h3>
          <p>
            Cross-references findings with medical knowledge bases, evaluates
            differential diagnoses, and assigns confidence levels to each
            hypothesis.
          </p>
        </div>

        <div className="agent-card">
          <div className="agent-card-icon report">📋</div>
          <h3>Reporting Agent</h3>
          <p>
            Compiles a structured diagnostic summary in clinical language with
            recommendations — fully editable by radiologists.
          </p>
        </div>
      </div>
    </main>
  );
}

export default LandingPage;
