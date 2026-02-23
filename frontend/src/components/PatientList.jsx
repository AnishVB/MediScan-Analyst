import { useState, useEffect, useCallback } from "react";
import AddPatientModal from "./AddPatientModal";

function PatientList({ selectedPatient, onSelectPatient, onViewProfile }) {
  const [patients, setPatients] = useState([]);
  const [search, setSearch] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);
  const [loading, setLoading] = useState(true);

  const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

  const fetchPatients = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/patients`);
      const data = await res.json();
      setPatients(data);
    } catch (err) {
      console.error("Failed to load patients", err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  const handlePatientAdded = (id) => {
    setShowAddModal(false);
    fetchPatients();
  };

  const filtered = patients.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="patient-list-section">
      <div className="uploader-title">Patient</div>

      <div className="patient-search-row">
        <input
          className="patient-search"
          type="text"
          placeholder="Search patients..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button
          className="btn-add-patient"
          onClick={() => setShowAddModal(true)}
        >
          +
        </button>
      </div>

      <div className="patient-list">
        {loading ? (
          <div className="patient-list-empty">Loading...</div>
        ) : filtered.length === 0 ? (
          <div className="patient-list-empty">
            {search ? "No patients match" : "No patients yet"}
          </div>
        ) : (
          filtered.map((p) => (
            <div
              key={p.id}
              className={`patient-item ${selectedPatient?.id === p.id ? "selected" : ""}`}
              onClick={() => onSelectPatient(p)}
            >
              <div className="patient-item-avatar">
                {p.name.charAt(0).toUpperCase()}
              </div>
              <div className="patient-item-info">
                <div className="patient-item-name">{p.name}</div>
                <div className="patient-item-meta">
                  {p.age ? `${p.age}y` : ""} {p.gender || ""}{" "}
                  {p.blood_group ? `• ${p.blood_group}` : ""}
                  {p.scan_count > 0 && (
                    <span className="scan-badge">{p.scan_count} scans</span>
                  )}
                </div>
              </div>
              <button
                className="patient-profile-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onViewProfile(p.id);
                }}
                title="View profile"
              >
                →
              </button>
            </div>
          ))
        )}
      </div>

      {showAddModal && (
        <AddPatientModal
          onClose={() => setShowAddModal(false)}
          onPatientAdded={handlePatientAdded}
        />
      )}
    </div>
  );
}

export default PatientList;
