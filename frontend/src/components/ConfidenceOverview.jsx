function ConfidenceOverview({ analysisResult }) {
  if (!analysisResult) return null;

  const mc = analysisResult.model_confidence || {};
  const classification = Math.round((mc.image_classification || 0) * 100);
  const findings = Math.round((mc.findings_extraction || 0) * 100);
  const dlConf = Math.round((mc.dl_classification || 0) * 100);
  const heuristicConf = Math.round((mc.heuristic_analysis || 0) * 100);
  const ensemble = Math.round((analysisResult.ensemble_confidence || 0) * 100);

  const dlAvailable = analysisResult.dl_predictions?.available;
  const modelsUsed = analysisResult.dl_predictions?.models_used || [];

  return (
    <div className="confidence-overview">
      <div className="panel-header">
        <h2>📊 Confidence Metrics</h2>
        {dlAvailable && (
          <span className="panel-badge dl-badge">🤖 DL Active</span>
        )}
      </div>
      <div className="confidence-grid">
        <div className="confidence-item">
          <div className="label">Classification</div>
          <div className={`value ${classification < 60 ? "warning" : ""}`}>
            {classification}%
          </div>
          <div className="conf-bar">
            <div
              className="conf-bar-fill"
              style={{ width: `${classification}%` }}
            />
          </div>
        </div>

        {dlAvailable && dlConf > 0 && (
          <div className="confidence-item dl-item">
            <div className="label">🤖 DL Model</div>
            <div className={`value ${dlConf < 40 ? "warning" : ""}`}>
              {dlConf}%
            </div>
            <div className="conf-bar">
              <div
                className="conf-bar-fill dl-fill"
                style={{ width: `${dlConf}%` }}
              />
            </div>
          </div>
        )}

        <div className="confidence-item">
          <div className="label">
            {dlAvailable ? "CV2 Heuristic" : "Findings"}
          </div>
          <div
            className={`value ${(dlAvailable ? heuristicConf : findings) < 60 ? "warning" : ""}`}
          >
            {dlAvailable ? heuristicConf : findings}%
          </div>
          <div className="conf-bar">
            <div
              className="conf-bar-fill heuristic-fill"
              style={{ width: `${dlAvailable ? heuristicConf : findings}%` }}
            />
          </div>
        </div>

        <div className="confidence-item">
          <div className="label">Ensemble</div>
          <div className={`value ${ensemble < 60 ? "warning" : ""}`}>
            {ensemble}%
          </div>
          <div className="conf-bar">
            <div
              className="conf-bar-fill ensemble-fill"
              style={{ width: `${ensemble}%` }}
            />
          </div>
        </div>
      </div>

      {modelsUsed.length > 0 && (
        <div className="models-used">
          <div className="models-used-title">Models Used</div>
          <div className="models-used-tags">
            {modelsUsed.map((model, i) => (
              <span key={i} className="model-tag">
                {model}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ConfidenceOverview;
