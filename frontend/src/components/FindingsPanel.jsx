function ConfidenceBar({ value }) {
  const percent = Math.round(value * 100);
  const level = percent >= 75 ? "high" : percent >= 50 ? "medium" : "low";
  return (
    <div className="finding-confidence">
      <div className="confidence-bar">
        <div
          className={`confidence-fill ${level}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="confidence-value">{percent}%</span>
    </div>
  );
}

function FindingsPanel({ findings }) {
  if (!findings || findings.length === 0) return null;

  return (
    <div className="findings-panel">
      <div className="panel-header">
        <h2>🔍 Findings</h2>
        <span className="panel-badge">{findings.length} detected</span>
      </div>

      {findings.map((finding, i) => (
        <div
          key={i}
          className="finding-card"
          style={{ animationDelay: `${i * 0.1}s` }}
        >
          <div className="finding-header">
            <span className="finding-type">{finding.finding_type}</span>
            <ConfidenceBar value={finding.confidence} />
          </div>
          <div className="finding-location">📍 {finding.location}</div>
          <div className="finding-desc">{finding.description}</div>
        </div>
      ))}
    </div>
  );
}

export default FindingsPanel;
