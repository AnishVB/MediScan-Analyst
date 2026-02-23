function ConfidenceOverview({ analysisResult }) {
  if (!analysisResult) return null;

  const mc = analysisResult.model_confidence || {};
  const classification = Math.round((mc.image_classification || 0) * 100);
  const findings = Math.round((mc.findings_extraction || 0) * 100);
  const ensemble = Math.round((analysisResult.ensemble_confidence || 0) * 100);

  return (
    <div className="confidence-overview">
      <div className="panel-header">
        <h2>📊 Confidence Metrics</h2>
      </div>
      <div className="confidence-grid">
        <div className="confidence-item">
          <div className="label">Classification</div>
          <div className={`value ${classification < 60 ? "warning" : ""}`}>
            {classification}%
          </div>
        </div>
        <div className="confidence-item">
          <div className="label">Findings</div>
          <div className={`value ${findings < 60 ? "warning" : ""}`}>
            {findings}%
          </div>
        </div>
        <div className="confidence-item">
          <div className="label">Ensemble</div>
          <div className={`value ${ensemble < 60 ? "warning" : ""}`}>
            {ensemble}%
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfidenceOverview;
