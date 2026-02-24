import { useState, useCallback } from "react";
import Header from "./components/Header";
import LandingPage from "./components/LandingPage";
import Dashboard from "./components/Dashboard";
import PatientProfile from "./components/PatientProfile";
import ToastContainer from "./components/ToastContainer";

const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000";

function App() {
  const [page, setPage] = useState("landing");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [agentStage, setAgentStage] = useState("idle");
  const [toasts, setToasts] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [profilePatientId, setProfilePatientId] = useState(null);

  const addToast = useCallback((message, type = "info") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const handleImageSelect = useCallback((file) => {
    setSelectedImage(file);
    setAnalysisResult(null);
    setAgentStage("idle");
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleRemoveImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    setAnalysisResult(null);
    setAgentStage("idle");
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);

    // Stage 1: Vision Agent
    setAgentStage("vision");
    addToast("🔬 Vision Agent analyzing image...", "info");

    await new Promise((r) => setTimeout(r, 800));

    // Stage 2: DL Classification Agent
    setAgentStage("dl_classification");
    addToast("🤖 DL Classification Agent running CNN models...", "info");

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      if (selectedPatient) {
        formData.append("patient_id", selectedPatient.id.toString());
      }

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Analysis failed");

      const data = await response.json();

      // Stage 3: Analysis Agent
      setAgentStage("analysis");
      addToast("🧠 Analysis Agent merging heuristic + DL findings...", "info");
      await new Promise((r) => setTimeout(r, 600));

      // Stage 4: Reporting Agent
      setAgentStage("reporting");
      addToast("📋 Reporting Agent compiling summary...", "info");
      await new Promise((r) => setTimeout(r, 500));

      setAnalysisResult(data);
      setAgentStage("complete");

      const dlCount = data.dl_predictions?.models_used?.length || 0;
      if (dlCount > 0) {
        addToast(
          `✅ Analysis complete — ${dlCount} DL model(s) used`,
          "success",
        );
      } else {
        addToast(
          "✅ Analysis complete (heuristic mode) — train models for DL",
          "success",
        );
      }
    } catch (err) {
      addToast("❌ Analysis failed: " + err.message, "error");
      setAgentStage("idle");
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedImage, selectedPatient, addToast]);

  const handleViewProfile = useCallback((patientId) => {
    setProfilePatientId(patientId);
    setPage("profile");
  }, []);

  const handleBackFromProfile = useCallback(() => {
    setProfilePatientId(null);
    setPage("dashboard");
  }, []);

  const handleNavigate = useCallback((target) => {
    setProfilePatientId(null);
    setPage(target);
  }, []);

  return (
    <>
      <div className="app-bg" />
      <Header page={page} onNavigate={handleNavigate} />

      {page === "landing" && (
        <LandingPage onGetStarted={() => handleNavigate("dashboard")} />
      )}

      {page === "dashboard" && (
        <Dashboard
          selectedImage={selectedImage}
          imagePreview={imagePreview}
          onImageSelect={handleImageSelect}
          onRemoveImage={handleRemoveImage}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
          agentStage={agentStage}
          analysisResult={analysisResult}
          addToast={addToast}
          selectedPatient={selectedPatient}
          onSelectPatient={setSelectedPatient}
          onViewProfile={handleViewProfile}
        />
      )}

      {page === "profile" && profilePatientId && (
        <PatientProfile
          patientId={profilePatientId}
          onBack={handleBackFromProfile}
        />
      )}

      <ToastContainer toasts={toasts} />
    </>
  );
}

export default App;
