import { useState } from "react";

function AddPatientModal({ onClose, onPatientAdded }) {
  const [form, setForm] = useState({
    name: "",
    age: "",
    gender: "",
    blood_group: "",
    medical_conditions: "",
    notes: "",
  });
  const [saving, setSaving] = useState(false);

  const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.name.trim()) return;

    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/api/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          age: form.age ? parseInt(form.age) : null,
        }),
      });
      const data = await res.json();
      onPatientAdded(data.id);
    } catch (err) {
      console.error("Failed to create patient", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Add New Patient</h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-group">
            <label>Full Name *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => handleChange("name", e.target.value)}
              placeholder="Patient full name"
              required
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                value={form.age}
                onChange={(e) => handleChange("age", e.target.value)}
                placeholder="Age"
                min="0"
                max="150"
              />
            </div>
            <div className="form-group">
              <label>Gender</label>
              <select
                value={form.gender}
                onChange={(e) => handleChange("gender", e.target.value)}
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label>Blood Group</label>
              <select
                value={form.blood_group}
                onChange={(e) => handleChange("blood_group", e.target.value)}
              >
                <option value="">Select</option>
                <option value="A+">A+</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B-">B-</option>
                <option value="AB+">AB+</option>
                <option value="AB-">AB-</option>
                <option value="O+">O+</option>
                <option value="O-">O-</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Known Medical Conditions</label>
            <textarea
              value={form.medical_conditions}
              onChange={(e) =>
                handleChange("medical_conditions", e.target.value)
              }
              placeholder="e.g. Diabetes, Hypertension, Asthma (comma separated)"
              rows={2}
            />
          </div>

          <div className="form-group">
            <label>Notes</label>
            <textarea
              value={form.notes}
              onChange={(e) => handleChange("notes", e.target.value)}
              placeholder="Additional notes about the patient"
              rows={2}
            />
          </div>

          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={saving || !form.name.trim()}
            >
              {saving ? "Saving..." : "+ Add Patient"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AddPatientModal;
