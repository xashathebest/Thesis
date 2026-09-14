const $ = (id) => document.getElementById(id);
const elements = {
  systemPill: $("system-pill"), cameraPill: $("camera-pill"), modelPill: $("model-pill"), trackerPill: $("tracker-pill"),
  inspectionState: $("inspection-state"), liveBadge: $("live-badge"), alert: $("alert"),
  start: $("start-button"), stop: $("stop-button"), reset: $("reset-button"), video: $("video-feed"),
  placeholder: $("video-placeholder"), placeholderTitle: $("placeholder-title"), placeholderCopy: $("placeholder-copy"),
  empty: $("result-empty"), active: $("result-active"), resultPanel: $("result-panel"), resultFish: $("result-fish"),
  decision: $("decision"), resultLabel: $("result-label"), confidence: $("confidence"), confidenceBar: $("confidence-bar"),
  trackChips: $("track-chips"), historyEmpty: $("history-empty"), historyWrap: $("history-table-wrap"), historyBody: $("history-body"),
  previewBanner: $("preview-banner"), countersPanel: $("counters-panel"), historyPanel: $("history-panel"),
  resultEmptyTitle: $("result-empty-title"), resultEmptyCopy: $("result-empty-copy"),
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
  element.replaceChildren(document.createElement("i"), document.createTextNode(text));
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

function renderTracks(tracks) {
  elements.trackChips.replaceChildren();
  if (!tracks.length) {
    const empty = document.createElement("span");
    empty.className = "track-empty";
    empty.textContent = "Waiting for tracked fish";
    elements.trackChips.append(empty);
    return;
  }
  tracks.slice(0, 4).forEach((track) => {
    const chip = document.createElement("span");
    chip.className = `track-chip ${track.current_class === "Rejected" ? "rejected" : ""}`;
    chip.textContent = `#${track.track_id} ${track.current_class} ${Number(track.confidence_percent).toFixed(1)}%`;
    elements.trackChips.append(chip);
  });
  if (tracks.length > 4) {
    const extra = document.createElement("span");
    extra.className = "track-empty";
    extra.textContent = `+${tracks.length - 4} more`;
    elements.trackChips.append(extra);
  }
}

function formatTime(timestamp) {
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.getTime()) ? timestamp : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function tableCell(label, value, className = "") {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  cell.className = className;
  cell.textContent = value;
  return cell;
}

function renderHistory(history) {
  elements.historyEmpty.classList.toggle("hidden", history.length > 0);
  elements.historyWrap.classList.toggle("hidden", history.length === 0);
  elements.historyBody.replaceChildren();
  history.forEach((event) => {
    const row = document.createElement("tr");
    row.append(
      tableCell("Fish ID", event.fish_label),
      tableCell("Time", formatTime(event.timestamp)),
      tableCell("Class", event.final_class),
      tableCell("Confidence", `${Number(event.confidence_percent).toFixed(1)}%`),
      tableCell("Decision", event.decision, `history-decision ${event.decision === "REJECTED" ? "rejected" : "accepted"}`),
    );
    elements.historyBody.append(row);
  });
}

function render(data) {
  const inspection = data.inspection_status;
  const partPreview = data.runtime_mode === "part_preview";
  elements.previewBanner.classList.toggle("hidden", !partPreview);
  elements.countersPanel.classList.toggle("preview-disabled", partPreview);
  elements.historyPanel.classList.toggle("preview-disabled", partPreview);
  elements.resultPanel.classList.toggle("preview-disabled", partPreview);
  $("active-fish-card").classList.toggle("preview-disabled", partPreview);
  $("total-events-card").classList.toggle("preview-disabled", partPreview);
  $("workspace-title").textContent = partPreview ? "Live part-model preview" : "Live quality inspection";
  $("feed-title").textContent = partPreview ? "Live raw part predictions" : "Live tracked feed";
  setPill(elements.systemPill, `System ${titleCase(data.system_status)}`, data.system_status);
  setPill(elements.cameraPill, `Camera ${titleCase(data.camera_status)}`, data.camera_status);
  setPill(elements.modelPill, `Model ${titleCase(data.model_status)}`, data.model_status);
  setPill(elements.trackerPill, partPreview ? "Fish tracker disabled" : `Tracker ${titleCase(data.tracker_status)}`, data.tracker_status);

  elements.inspectionState.className = `inspection-state ${stateClass(inspection)}`;
  elements.inspectionState.replaceChildren(document.createElement("i"), document.createTextNode(titleCase(inspection)));
  elements.liveBadge.className = `live-badge ${stateClass(inspection)}`;
  elements.liveBadge.textContent = inspection === "running" ? "LIVE" : inspection.toUpperCase();
  elements.start.disabled = ["starting", "running"].includes(inspection) || data.model_status !== "ready";
  elements.stop.disabled = !["starting", "running"].includes(inspection);
  elements.reset.disabled = inspection === "starting";

  if (["starting", "running"].includes(inspection)) {
    attachStream();
  } else if (streamAttached) {
    elements.video.removeAttribute("src");
    streamAttached = false;
    streamReady = false;
  }
  elements.placeholder.classList.toggle("hidden", inspection === "running" && streamReady);
  if (inspection === "starting") {
    elements.placeholderTitle.textContent = partPreview ? "Opening camera and part model" : "Opening camera and tracker";
    elements.placeholderCopy.textContent = partPreview
      ? "The local raw part-segmentation preview is starting."
      : "The local ByteTrack inspection worker is starting.";
  } else if (inspection === "errored") {
    elements.placeholderTitle.textContent = "Inspection unavailable";
    elements.placeholderCopy.textContent = data.message;
  } else {
    elements.placeholderTitle.textContent = "Inspection stopped";
    elements.placeholderCopy.textContent = "Start inspection to open the local camera.";
  }

  const showAlert = data.model_status !== "ready" || inspection === "errored" || data.tracker_status === "error";
  elements.alert.classList.toggle("hidden", !showAlert);
  elements.alert.textContent = data.message;

  const event = partPreview ? null : data.latest_event;
  elements.empty.classList.toggle("hidden", Boolean(event));
  elements.active.classList.toggle("hidden", !event);
  elements.resultPanel.classList.toggle("is-rejected", event?.decision === "REJECTED");
  $("result-eyebrow").textContent = partPreview ? "Preview mode" : "Latest completed inspection";
  $("result-title").textContent = partPreview ? "Whole-fish result unavailable" : "Current classification event";
  elements.resultEmptyTitle.textContent = partPreview ? "Unavailable in Part Preview" : "No completed inspection";
  elements.resultEmptyCopy.textContent = partPreview
    ? "Raw part predictions are never converted into fish grades or inspection events."
    : "A result will appear after a tracked fish crosses the inspection line.";
  $("classification-policy").textContent = partPreview
    ? "Head, Body, and Tail labels are preview evidence only."
    : 'Class A-C are accepted. "Rejected" is rejected.';
  if (event) {
    elements.decision.textContent = event.decision;
    elements.decision.className = `decision ${event.decision === "REJECTED" ? "rejected" : ""}`;
    elements.resultFish.textContent = event.fish_label;
    elements.resultLabel.textContent = event.final_class;
    elements.confidence.textContent = `${Number(event.confidence_percent).toFixed(1)}%`;
    elements.confidenceBar.style.width = `${Math.min(100, event.confidence_percent)}%`;
  }

  const fps = Number(data.fps || 0);
  $("feed-fps").textContent = `${fps.toFixed(1)} FPS`;
  $("fps-metric").textContent = `${fps.toFixed(1)} FPS`;
  $("camera-metric").textContent = titleCase(data.camera_status);
  $("model-metric").textContent = titleCase(data.model_status);
  $("model-name").textContent = data.active_model || "No model loaded";
  $("active-fish-count").textContent = partPreview ? "N/A" : data.active_fish_count;
  $("active-fish-inline").textContent = partPreview ? "N/A" : data.active_fish_count;
  $("tracker-summary").textContent = partPreview
    ? "Disabled in Part Preview"
    : `${titleCase(data.tracking_config.tracker.replace(".yaml", ""))} ${titleCase(data.tracker_status)}`;
  $("threshold").textContent = `${Math.round(data.confidence_threshold * 100)}%`;
  $("total-count").textContent = partPreview ? "N/A" : data.counters.total;
  $("class-a-count").textContent = partPreview ? "—" : data.counters["Class A"];
  $("class-b-count").textContent = partPreview ? "—" : data.counters["Class B"];
  $("class-c-count").textContent = partPreview ? "—" : data.counters["Class C"];
  $("rejected-count").textContent = partPreview ? "—" : data.counters.Rejected;
  $("history-limit").textContent = partPreview ? "Disabled" : `Newest ${data.tracking_config.history_limit} events`;
  $("line-description").textContent = partPreview
    ? "Raw 12-class part segmentation - no tracking"
    : `${titleCase(data.tracking_config.conveyor_direction)} - ${Math.round(data.tracking_config.line_position * 100)}% line`;
  if (partPreview) {
    elements.trackChips.replaceChildren();
    const notice = document.createElement("span");
    notice.className = "track-empty";
    notice.textContent = "No fish IDs in Part Preview";
    elements.trackChips.append(notice);
    elements.historyEmpty.classList.remove("hidden");
    elements.historyEmpty.textContent = "Unavailable in Part Preview — no inspection events are created.";
    elements.historyWrap.classList.add("hidden");
    elements.historyBody.replaceChildren();
  } else {
    elements.historyEmpty.textContent = "No fish have crossed the inspection line in this session.";
    renderTracks(data.active_tracks);
    renderHistory(data.recent_history);
  }
}

function renderBackendUnavailable() {
  setPill(elements.systemPill, "Backend unavailable", "error");
  elements.alert.classList.remove("hidden");
  elements.alert.textContent = "Cannot reach the local Python backend. Check that it is running, then reload this page.";
  elements.start.disabled = true;
  elements.stop.disabled = true;
  elements.reset.disabled = true;
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

async function postControl(path) {
  elements.start.disabled = true;
  elements.stop.disabled = true;
  elements.reset.disabled = true;
  try {
    const response = await fetch(path, { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (_) {
    renderBackendUnavailable();
  }
}

elements.start.addEventListener("click", () => postControl("/api/inspection/start"));
elements.stop.addEventListener("click", () => postControl("/api/inspection/stop"));
elements.reset.addEventListener("click", () => postControl("/api/session/reset"));
getStatus();
setInterval(getStatus, 700);
