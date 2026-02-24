const AGENTS = [
  {
    id: "vision",
    name: "Vision Agent",
    icon: "👁️",
    desc: "OpenCV image processing, anatomical structure detection",
  },
  {
    id: "dl_classification",
    name: "DL Classification",
    icon: "🤖",
    desc: "CNN pathology detection (DenseNet-121 + Custom MediScanCNN)",
  },
  {
    id: "analysis",
    name: "Analysis Agent",
    icon: "🧠",
    desc: "Merges heuristic + DL findings, evaluates differential diagnoses",
  },
  {
    id: "reporting",
    name: "Reporting Agent",
    icon: "📋",
    desc: "Compiles structured diagnostic summary with recommendations",
  },
];

function getStatus(agentId, currentStage) {
  const order = [
    "idle",
    "vision",
    "dl_classification",
    "analysis",
    "reporting",
    "complete",
  ];
  const agentIndex = order.indexOf(agentId);
  const currentIndex = order.indexOf(currentStage);

  if (currentStage === "idle") return "waiting";
  if (currentStage === "complete") return "complete";
  if (agentIndex === currentIndex) return "processing";
  if (agentIndex < currentIndex) return "complete";
  return "waiting";
}

function AgentPipeline({ agentStage }) {
  return (
    <div className="pipeline">
      <div className="pipeline-title">Agent Pipeline</div>
      <div className="pipeline-agents">
        {AGENTS.map((agent, i) => {
          const status = getStatus(agent.id, agentStage);
          return (
            <div key={agent.id} className="pipeline-agent">
              <div className="agent-indicator">
                <div
                  className={`agent-dot ${status === "processing" ? "running" : status === "complete" ? "done" : "idle"}`}
                >
                  {status === "complete" ? "✓" : agent.icon}
                </div>
                {i < AGENTS.length - 1 && (
                  <div
                    className={`agent-connector ${status === "complete" ? "active" : ""}`}
                  />
                )}
              </div>
              <div className="agent-info">
                <h4>{agent.name}</h4>
                <div className="agent-desc">{agent.desc}</div>
                <div className={`agent-status ${status}`}>
                  {status === "processing" && <span className="spinner" />}
                  {status === "waiting" && "Waiting"}
                  {status === "processing" && "Processing..."}
                  {status === "complete" && "✓ Complete"}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default AgentPipeline;
