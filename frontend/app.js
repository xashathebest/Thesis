const $ = (id) => document.getElementById(id);
const elements = {
  systemPill: $("system-pill"), cameraPill: $("camera-pill"), modelPill: $("model-pill"),
  inspectionState: $("inspection-state"), liveBadge: $("live-badge"), alert: $("alert"),
  start: $("start-button"), stop: $("stop-button"), video: $("video-feed"),
  placeholder: $("video-placeholder"), placeholderTitle: $("placeholder-title"), placeholderCopy: $("placeholder-copy"),
  empty: $("result-empty"), active: $("result-active"), resultPanel: $("result-panel"),
  decision: $("decision"), resultLabel: $("result-label"), confidence: $("confidence"), confidenceBar: $("confidence-bar"),
};

let streamAttached = false;
let streamReady = false;

function titleCase(value) {
  return (value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function stateClass(value) {
  if (["healthy", "ready", "connected", "running"].includes(value)) return "healthy";
  if (["starting", "connecting", "checking", "attention"].includes(value)) return "warning";
  if (["errored", "error", "unavailable"].includes(value)) return "error";
  return "neutral";
}

function setPill(element, text, state) {
  element.className = `status-pill ${stateClass(state)}`;
  element.innerHTML = `<i></i>${text}`;
}

function attachStream() {
  if (streamAttached) return;
  elements.video.src = `/api/video-feed?t=${Date.now()}`;
  streamAttached = true;
}

elements.video.addEventListener("load", () => {
  streamReady = true;
  elements.placeholder.classList.add("hidden");
});

function render(data) {
  const inspection = data.inspection_status;
  setPill(elements.systemPill, `System ${titleCase(data.system_status)}`, data.system_status);
  setPill(elements.cameraPill, `Camera ${titleCase(data.camera_status)}`, data.camera_status);
  setPill(elements.modelPill, `Model ${titleCase(data.model_status)}`, data.model_status);

  elements.inspectionState.className = `inspection-state ${stateClass(inspection)}`;
  elements.inspectionState.innerHTML = `<i></i>${titleCase(inspection)}`;
  elements.liveBadge.className = `live-badge ${stateClass(inspection)}`;
  elements.liveBadge.textContent = inspection === "running" ? "LIVE" : inspection.toUpperCase();
  elements.start.disabled = ["starting", "running"].includes(inspection) || data.model_status !== "ready";
  elements.stop.disabled = !["starting", "running"].includes(inspection);

  if (["starting", "running"].includes(inspection)) {
    attachStream();
  } else if (streamAttached) {
    elements.video.removeAttribute("src");
    streamAttached = false;
    streamReady = false;
  }
  const showVideo = inspection === "running" && streamReady;
  elements.placeholder.classList.toggle("hidden", showVideo);
  if (inspection === "starting") {
    elements.placeholderTitle.textContent = "Opening camera";
    elements.placeholderCopy.textContent = "The local inspection worker is starting.";
  } else if (inspection === "errored") {
    elements.placeholderTitle.textContent = "Inspection unavailable";
    elements.placeholderCopy.textContent = data.message;
  } else {
    elements.placeholderTitle.textContent = "Inspection stopped";
    elements.placeholderCopy.textContent = "Start inspection to open the local camera.";
  }

  const showAlert = data.model_status !== "ready" || inspection === "errored";
  elements.alert.classList.toggle("hidden", !showAlert);
  elements.alert.textContent = data.message;

  const detection = data.current_detection;
  elements.empty.classList.toggle("hidden", Boolean(detection));
  elements.active.classList.toggle("hidden", !detection);
  elements.resultPanel.classList.toggle("is-rejected", detection?.decision === "REJECTED");
  if (detection) {
    elements.decision.textContent = detection.decision;
    elements.decision.className = `decision ${detection.decision === "REJECTED" ? "rejected" : ""}`;
    elements.resultLabel.textContent = detection.class_name;
    elements.confidence.textContent = `${detection.confidence_percent.toFixed(1)}%`;
    elements.confidenceBar.style.width = `${Math.min(100, detection.confidence_percent)}%`;
  }

  $("feed-fps").textContent = `${data.fps.toFixed(1)} FPS`;
  $("fps-metric").textContent = `${data.fps.toFixed(1)} FPS`;
  $("camera-metric").textContent = titleCase(data.camera_status);
  $("model-metric").textContent = titleCase(data.model_status);
  $("model-name").textContent = data.active_model || "No model loaded";
  $("threshold").textContent = `${Math.round(data.confidence_threshold * 100)}%`;
  $("total-count").textContent = data.counters.total;
  $("class-a-count").textContent = data.counters["Class A"];
  $("class-b-count").textContent = data.counters["Class B"];
  $("class-c-count").textContent = data.counters["Class C"];
  $("rejected-count").textContent = data.counters.Rejected;
}

function renderBackendUnavailable() {
  setPill(elements.systemPill, "Backend unavailable", "error");
  elements.alert.classList.remove("hidden");
  elements.alert.textContent = "Cannot reach the local Python backend. Check that it is running, then reload this page.";
  elements.start.disabled = true;
  elements.stop.disabled = true;
}

async function getStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (_) {
    renderBackendUnavailable();
  }
}

async function control(action) {
  elements.start.disabled = true;
  elements.stop.disabled = true;
  try {
    const response = await fetch(`/api/inspection/${action}`, { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (_) {
    renderBackendUnavailable();
  }
}

elements.start.addEventListener("click", () => control("start"));
elements.stop.addEventListener("click", () => control("stop"));
getStatus();
setInterval(getStatus, 700);
