import { useState, useEffect, useCallback } from "react";

function PatientProfile({ patientId, onBack }) {
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedScan, setExpandedScan] = useState(null);

  const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

  const fetchProfile = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/patients/${patientId}`);
      const data = await res.json();
      setPatient(data);
    } catch (err) {
      console.error("Failed to load patient", err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE, patientId]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  if (loading) {
    return (
      <div className="empty-state">
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <h3>Loading patient profile...</h3>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="empty-state">
        <h3>Patient not found</h3>
        <button className="btn-secondary" onClick={onBack}>
          ← Back
        </button>
      </div>
    );
  }

  const scans = patient.scans || [];
  const referrals = patient.referrals || [];

  return (
    <div className="profile-page">
      <button className="btn-back" onClick={onBack}>
        ← Back to Dashboard
      </button>

      {/* Patient Header */}
      <div className="profile-header">
        <div className="profile-avatar">
          {patient.name.charAt(0).toUpperCase()}
        </div>
        <div className="profile-info">
          <h1>{patient.name}</h1>
          <div className="profile-meta">
            {patient.age && <span>{patient.age} years old</span>}
            {patient.gender && <span>{patient.gender}</span>}
            {patient.blood_group && <span>Blood: {patient.blood_group}</span>}
          </div>
          {patient.medical_conditions && (
            <div className="profile-conditions">
              <strong>Conditions:</strong> {patient.medical_conditions}
            </div>
          )}
          {patient.notes && (
            <div className="profile-notes">
              <strong>Notes:</strong> {patient.notes}
            </div>
          )}
        </div>
        <div className="profile-stats">
          <div className="profile-stat">
            <div className="profile-stat-value">{scans.length}</div>
            <div className="profile-stat-label">Scans</div>
          </div>
          <div className="profile-stat">
            <div className="profile-stat-value">{referrals.length}</div>
            <div className="profile-stat-label">Referrals</div>
          </div>
        </div>
      </div>

      {/* Scan History */}
      <div className="profile-section">
        <h2>📊 Scan History</h2>
        {scans.length === 0 ? (
          <div className="profile-empty">No scans recorded yet</div>
        ) : (
          <div className="scan-timeline">
            {scans.map((scan) => {
              const analysis = scan.analysis_json;
              const findings = analysis?.analysis?.findings || [];
              const isExpanded = expandedScan === scan.id;

              return (
                <div key={scan.id} className="scan-timeline-item">
                  <div className="scan-timeline-dot" />
                  <div
                    className="scan-timeline-card"
                    onClick={() => setExpandedScan(isExpanded ? null : scan.id)}
                  >
                    <div className="scan-timeline-header">
                      <div>
                        <span className="scan-type-badge">
                          {scan.image_type?.toUpperCase()}
                        </span>
                        <span className="scan-body-part">{scan.body_part}</span>
                      </div>
                      <div className="scan-date">
                        {new Date(scan.created_at).toLocaleDateString()}
                      </div>
                    </div>

                    {scan.confidence && (
                      <div className="scan-confidence-mini">
                        Confidence: {Math.round(scan.confidence * 100)}%
                      </div>
                    )}

                    {isExpanded && (
                      <div className="scan-expanded">
                        <div className="scan-expanded-title">Findings:</div>
                        {findings.map((f, i) => (
                          <div key={i} className="scan-finding-mini">
                            <strong>{f.finding_type}</strong>: {f.description}
                            <span className="scan-finding-conf">
                              {Math.round(f.confidence * 100)}%
                            </span>
                          </div>
                        ))}

                        {analysis?.analysis?.primary_hypothesis && (
                          <div className="scan-primary-mini">
                            <strong>Assessment:</strong>{" "}
                            {analysis.analysis.primary_hypothesis.condition}
                            <span
                              className={`report-tag risk-${analysis.analysis.primary_hypothesis.risk_level?.toLowerCase()}`}
                            >
                              {analysis.analysis.primary_hypothesis.risk_level}
                            </span>
                          </div>
                        )}

                        {analysis?.analysis?.recommendations && (
                          <div className="scan-recs-mini">
                            <strong>Recommendations:</strong>
                            <ul>
                              {analysis.analysis.recommendations.map((r, j) => (
                                <li key={j}>{r}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    <div className="scan-expand-hint">
                      {isExpanded ? "▲ Collapse" : "▼ View details"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Referrals */}
      {referrals.length > 0 && (
        <div className="profile-section">
          <h2>📨 Referrals</h2>
          <div className="referral-list">
            {referrals.map((ref) => (
              <div key={ref.id} className="referral-card">
                <div className="referral-header">
                  <span className="referral-specialist">
                    {ref.specialist_name}
                  </span>
                  <span
                    className={`referral-status status-${ref.status?.toLowerCase()}`}
                  >
                    {ref.status}
                  </span>
                </div>
                <div className="referral-field">{ref.specialist_field}</div>
                {ref.notes && <div className="referral-notes">{ref.notes}</div>}
                <div className="referral-date">
                  {new Date(ref.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PatientProfile;
