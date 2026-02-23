import ImageUploader from "./ImageUploader";
import AgentPipeline from "./AgentPipeline";
import PatientList from "./PatientList";
import FindingsPanel from "./FindingsPanel";
import DiagnosticReport from "./DiagnosticReport";
import ConfidenceOverview from "./ConfidenceOverview";

function Dashboard({
  selectedImage,
  imagePreview,
  onImageSelect,
  onRemoveImage,
  onAnalyze,
  isAnalyzing,
  agentStage,
  analysisResult,
  addToast,
  selectedPatient,
  onSelectPatient,
  onViewProfile,
}) {
  return (
    <div className="dashboard">
      {/* Sidebar */}
      <aside className="sidebar">
        <PatientList
          selectedPatient={selectedPatient}
          onSelectPatient={onSelectPatient}
          onViewProfile={onViewProfile}
        />

        <ImageUploader
          selectedImage={selectedImage}
          imagePreview={imagePreview}
          onImageSelect={onImageSelect}
          onRemoveImage={onRemoveImage}
          onAnalyze={onAnalyze}
          isAnalyzing={isAnalyzing}
        />

        <AgentPipeline agentStage={agentStage} />

        {analysisResult && (
          <div className="image-info">
            <div className="uploader-title">Image Details</div>
            <div className="info-row">
              <span className="info-label">Type</span>
              <span className="info-value">
                {analysisResult.image_type?.toUpperCase()}
              </span>
            </div>
            <div className="info-row">
              <span className="info-label">Body Part</span>
              <span className="info-value">{analysisResult.body_part}</span>
            </div>
            <div className="info-row">
              <span className="info-label">Processing Time</span>
              <span className="info-value">
                {analysisResult.processing_time?.toFixed(2)}s
              </span>
            </div>
            <div className="info-row">
              <span className="info-label">Models Used</span>
              <span className="info-value">
                {analysisResult.models_used?.length || 0}
              </span>
            </div>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="main-content">
        {selectedPatient && (
          <div className="selected-patient-banner">
            <div className="selected-patient-info">
              <span className="patient-banner-avatar">
                {selectedPatient.name.charAt(0)}
              </span>
              <div>
                <strong>{selectedPatient.name}</strong>
                <span className="patient-banner-meta">
                  {selectedPatient.age ? `${selectedPatient.age}y` : ""}{" "}
                  {selectedPatient.gender || ""}
                  {selectedPatient.medical_conditions
                    ? ` — ${selectedPatient.medical_conditions}`
                    : ""}
                </span>
              </div>
            </div>
            <button
              className="btn-remove"
              onClick={() => onSelectPatient(null)}
            >
              ✕ Deselect
            </button>
          </div>
        )}

        {!analysisResult && agentStage === "idle" ? (
          <div className="empty-state">
            <div className="empty-state-icon">🏥</div>
            <h3>Ready for Analysis</h3>
            <p>
              {selectedPatient
                ? `Select or upload a medical image for ${selectedPatient.name}'s analysis.`
                : "Select a patient and upload a medical image to begin the multi-agent diagnostic pipeline."}
            </p>
          </div>
        ) : !analysisResult ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <span
                className="spinner"
                style={{ width: 48, height: 48, borderWidth: 3 }}
              />
            </div>
            <h3>Agents Working...</h3>
            <p>
              The multi-agent pipeline is processing your medical image. Please
              wait.
            </p>
          </div>
        ) : (
          <>
            <ConfidenceOverview analysisResult={analysisResult} />

            <div className="results-grid">
              <FindingsPanel findings={analysisResult.analysis?.findings} />
              <DiagnosticReport
                analysisResult={analysisResult}
                addToast={addToast}
                selectedPatient={selectedPatient}
              />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default Dashboard;
