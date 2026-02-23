import { useState } from "react";

const SPECIALIST_FIELDS = [
  "Radiology",
  "Pulmonology",
  "Cardiology",
  "Neurology",
  "Orthopedics",
  "Oncology",
  "General Medicine",
  "Surgery",
  "Other",
];

function ReferralModal({ patientId, scanId, onClose, onSent }) {
  const [form, setForm] = useState({
    specialist_name: "",
    specialist_field: "",
    notes: "",
  });
  const [sending, setSending] = useState(false);

  const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.specialist_name.trim() || !form.specialist_field) return;

    setSending(true);
    try {
      const res = await fetch(`${API_BASE}/api/referrals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          scan_id: scanId || null,
          ...form,
        }),
      });
      const data = await res.json();
      onSent(data);
    } catch (err) {
      console.error("Failed to send referral", err);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>📨 Send to Specialist</h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-group">
            <label>Specialist Name *</label>
            <input
              type="text"
              value={form.specialist_name}
              onChange={(e) =>
                setForm({ ...form, specialist_name: e.target.value })
              }
              placeholder="Dr. Smith"
              required
            />
          </div>

          <div className="form-group">
            <label>Field / Department *</label>
            <select
              value={form.specialist_field}
              onChange={(e) =>
                setForm({ ...form, specialist_field: e.target.value })
              }
              required
            >
              <option value="">Select specialty</option>
              {SPECIALIST_FIELDS.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Notes / Reason for Referral</label>
            <textarea
              value={form.notes}
              onChange={(e) => setForm({ ...form, notes: e.target.value })}
              placeholder="Include relevant clinical context, urgency level, specific questions..."
              rows={4}
            />
          </div>

          <div className="referral-summary">
            <div className="report-section-title">What will be sent:</div>
            <ul className="report-recommendations">
              <li>Complete analysis report with findings</li>
              <li>Confidence metrics and differential diagnoses</li>
              <li>Patient medical history summary</li>
              <li>Original scan image reference</li>
            </ul>
          </div>

          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={
                sending ||
                !form.specialist_name.trim() ||
                !form.specialist_field
              }
            >
              {sending ? "Sending..." : "📨 Send Referral"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ReferralModal;
