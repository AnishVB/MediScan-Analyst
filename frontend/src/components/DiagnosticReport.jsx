import { useState, useCallback } from "react";
import ReferralModal from "./ReferralModal";

function DiagnosticReport({ analysisResult, addToast, selectedPatient }) {
  const [approved, setApproved] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showReferral, setShowReferral] = useState(false);
  const [savedScanId, setSavedScanId] = useState(null);
  const analysis = analysisResult?.analysis;

  const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

  const handleApprove = useCallback(() => {
    setApproved(true);
    addToast("✅ Report approved by radiologist", "success");
  }, [addToast]);

  const handleReject = useCallback(() => {
    addToast("⚠️ Report flagged for revision", "error");
  }, [addToast]);

  const handleExport = useCallback(() => {
    const report = {
      timestamp: new Date().toISOString(),
      patient: selectedPatient
        ? { id: selectedPatient.id, name: selectedPatient.name }
        : null,
      image_type: analysisResult.image_type,
      body_part: analysisResult.body_part,
      confidence: analysisResult.ensemble_confidence,
      findings: analysis.findings,
      primary_hypothesis: analysis.primary_hypothesis,
      recommendations: analysis.recommendations,
      status: approved ? "Approved" : "Pending Review",
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mediscan-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    addToast("📄 Report exported successfully", "success");
  }, [analysisResult, analysis, approved, selectedPatient, addToast]);

  const handleSaveToPatient = useCallback(async () => {
    if (!selectedPatient) {
      addToast("⚠️ Please select a patient first", "error");
      return;
    }
    setSaving(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/patients/${selectedPatient.id}/scans`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: selectedPatient.id,
            image_filename: analysisResult.image_filename || "",
            image_type: analysisResult.image_type,
            body_part: analysisResult.body_part,
            analysis_json: {
              analysis: analysisResult.analysis,
              report: analysisResult.report,
              model_confidence: analysisResult.model_confidence,
            },
            confidence: analysisResult.ensemble_confidence,
          }),
        },
      );
      const data = await res.json();
      setSaved(true);
      setSavedScanId(data.scan_id);
      addToast(`💾 Scan saved to ${selectedPatient.name}'s profile`, "success");
    } catch (err) {
      addToast("❌ Failed to save scan", "error");
    } finally {
      setSaving(false);
    }
  }, [selectedPatient, analysisResult, addToast, API_BASE]);

  const handleReferralSent = useCallback(
    (data) => {
      setShowReferral(false);
      addToast(
        `📨 Referral sent successfully (ID: ${data.referral_id})`,
        "success",
      );
    },
    [addToast],
  );

  if (!analysis) return null;

  const riskLevel = analysis.primary_hypothesis?.risk_level || "Unknown";
  const riskClass = riskLevel.toLowerCase().includes("high")
    ? "risk-high"
    : riskLevel.toLowerCase().includes("low")
      ? "risk-low"
      : "risk-medium";

  return (
    <div className="report-panel">
      <div className="panel-header">
        <h2>📋 Diagnostic Report</h2>
        <span className="panel-badge">
          {approved ? "✓ Approved" : "Pending Review"}
        </span>
      </div>

      <div className="disclaimer">
        ⚠️ This is an AI-generated preliminary report. All findings must be
        verified by a qualified radiologist before clinical use.
      </div>

      {analysis.patient_context && (
        <div className="patient-context-box" style={{ marginTop: "12px" }}>
          <div className="report-section-title">
            Patient Context (AI-Integrated)
          </div>
          <div className="report-text">{analysis.patient_context}</div>
        </div>
      )}

      <div className="report-section" style={{ marginTop: "20px" }}>
        <div className="report-section-title">Image Classification</div>
        <div className="report-text">
          <strong>Type:</strong> {analysisResult.image_type?.toUpperCase()} —{" "}
          <strong>Body Part:</strong> {analysisResult.body_part}
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">Primary Assessment</div>
        <div
          className="report-text"
          contentEditable
          suppressContentEditableWarning
        >
          {analysis.primary_hypothesis?.condition || "No assessment available"}
        </div>
        <div className="report-tags" style={{ marginTop: "12px" }}>
          <span className={`report-tag ${riskClass}`}>Risk: {riskLevel}</span>
          <span className="report-tag">
            Confidence:{" "}
            {Math.round((analysis.primary_hypothesis?.probability || 0) * 100)}%
          </span>
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">Reasoning</div>
        <div
          className="report-text"
          contentEditable
          suppressContentEditableWarning
        >
          {analysis.primary_hypothesis?.reasoning || "N/A"}
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">Clinical Significance</div>
        <div
          className="report-text"
          contentEditable
          suppressContentEditableWarning
        >
          {analysis.clinical_significance || "N/A"}
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">Recommendations</div>
        <ul className="report-recommendations">
          {(analysis.recommendations || []).map((rec, i) => (
            <li key={i}>{rec}</li>
          ))}
        </ul>
      </div>

      <div className="report-actions">
        <button
          className="btn-approve"
          onClick={handleApprove}
          disabled={approved}
        >
          {approved ? "✓ Approved" : "✓ Approve Report"}
        </button>
        <button className="btn-reject" onClick={handleReject}>
          ✕ Flag for Revision
        </button>
        <button className="btn-export" onClick={handleExport}>
          ↓ Export
        </button>
      </div>

      {/* Save to Patient & Send to Specialist */}
      <div className="report-actions-extra">
        <button
          className="btn-save-patient"
          onClick={handleSaveToPatient}
          disabled={saved || saving || !selectedPatient}
          title={!selectedPatient ? "Select a patient first" : ""}
        >
          {saving
            ? "⏳ Saving..."
            : saved
              ? "✓ Saved to Patient"
              : "💾 Save to Patient Profile"}
        </button>
        <button
          className="btn-send-specialist"
          onClick={() => setShowReferral(true)}
          disabled={!selectedPatient}
          title={!selectedPatient ? "Select a patient first" : ""}
        >
          📨 Send to Specialist
        </button>
      </div>

      {showReferral && selectedPatient && (
        <ReferralModal
          patientId={selectedPatient.id}
          scanId={savedScanId}
          onClose={() => setShowReferral(false)}
          onSent={handleReferralSent}
        />
      )}
    </div>
  );
}

export default DiagnosticReport;
