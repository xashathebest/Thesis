const $ = (id) => document.getElementById(id);

const elements = {
  shell: $("app-shell"), sidebarToggle: $("sidebar-toggle"),
  pageTitle: $("page-title"), pageSubtitle: $("page-subtitle"),
  cameraPill: $("camera-pill"), modelPill: $("model-pill"), devicePill: $("device-pill"),
  alert: $("alert"), alertTitle: $("alert-title"), alertCopy: $("alert-copy"), alertDetails: $("alert-details"),
  liveBadge: $("live-badge"), inspectionState: $("inspection-state"), start: $("start-button"), emptyStart: $("empty-start-button"), stop: $("stop-button"), reset: $("reset-button"),
  video: $("video-feed"), placeholder: $("video-placeholder"), placeholderTitle: $("placeholder-title"), placeholderCopy: $("placeholder-copy"),
  feedFps: $("feed-fps"), feedResolution: $("feed-resolution"),
  resultEmpty: $("result-empty"), resultActive: $("result-active"), resultEmptyTitle: $("result-empty-title"), resultEmptyCopy: $("result-empty-copy"),
  decision: $("decision"), resultFish: $("result-fish"), resultLabel: $("result-label"), confidence: $("confidence"), confidenceBar: $("confidence-bar"), resultProcessing: $("result-processing"), resultTime: $("result-time"), latestFeatures: $("latest-feature-list"), classificationPolicy: $("classification-policy"),
  latestVerdictDetails: $("latest-verdict-details"), latestVerdictStatus: $("latest-verdict-status"), latestVerdictReason: $("latest-verdict-reason"), latestDetectionConfidence: $("latest-detection-confidence"), latestModel2Evidence: $("latest-model2-evidence"), latestCoverage: $("latest-coverage"), latestEffectiveWeights: $("latest-effective-weights"),
  uploadMode: $("upload-mode-button"), closeUpload: $("close-upload-button"), uploadWorkspace: $("upload-workspace"), uploadInput: $("upload-input"), analyze: $("analyze-button"), uploadStatus: $("upload-status"), uploadNote: $("upload-note"), uploadImage: $("upload-result-image"), uploadResults: $("upload-results"), uploadFishCount: $("upload-fish-count"), uploadCountLabel: $("upload-count-label"),
  kpiGrid: $("kpi-grid"), featureCounterGrid: $("feature-counter-grid"), threshold: $("threshold"), openConfidence: $("open-confidence-button"),
  gradeChartEmpty: $("grade-chart-empty"), gradeChart: $("grade-chart"), gradeDonut: $("grade-donut"), gradeTotal: $("grade-total"), gradeLegend: $("grade-legend"),
  featureChartEmpty: $("feature-chart-empty"), featureChart: $("feature-chart"), throughputChartEmpty: $("throughput-chart-empty"), throughputChart: $("throughput-chart"), throughputLine: $("throughput-line"), throughputCaption: $("throughput-caption"),
  confidenceChartEmpty: $("confidence-chart-empty"), confidenceChart: $("confidence-chart"), averageGradeConfidence: $("average-grade-confidence"),
  recentEmpty: $("recent-empty"), recentWrap: $("recent-table-wrap"), recentHistoryBody: $("recent-history-body"), viewAll: $("view-all-button"),
  historyEmpty: $("history-empty"), historyWrap: $("history-table-wrap"), historyBody: $("history-body"), historySearch: $("history-search"), historyGrade: $("history-grade"), historyFeature: $("history-feature"), historyConfidence: $("history-confidence"), historyFilter: $("history-filter-button"), historyPrev: $("history-prev"), historyNext: $("history-next"), historyPage: $("history-page"),
  reviewEmpty: $("review-empty"), reviewWrap: $("review-table-wrap"), reviewBody: $("review-body"), reviewPrev: $("review-prev"), reviewNext: $("review-next"), reviewPage: $("review-page"),
  exportButton: $("export-button"), sidebarExport: $("sidebar-export"), exportModal: $("export-modal"), exportForm: $("export-form"), closeExport: $("close-export-button"), cancelExport: $("cancel-export-button"), dateRange: $("date-range"), exportStartDate: $("export-start-date"), exportEndDate: $("export-end-date"),
  settingsButton: $("settings-button"), sidebarSettings: $("sidebar-settings"), settingsDrawer: $("settings-drawer"), closeSettings: $("close-settings-button"), scrim: $("scrim"),
  cameraSettingsButton: $("camera-settings-button"), cameraCalibrationDrawer: $("camera-calibration-drawer"), closeCameraCalibration: $("close-camera-calibration-button"),
  cameraDeviceName: $("camera-device-name"), cameraConnection: $("camera-connection"), cameraLockBadge: $("camera-lock-badge"), cameraCaptureResolution: $("camera-capture-resolution"), cameraProcessingResolution: $("camera-processing-resolution"), cameraHardwareFps: $("camera-hardware-fps"), cameraBackend: $("camera-backend"), cameraProfileName: $("camera-profile-name"), cameraLockState: $("camera-lock-state"),
  calibrationMode: $("calibration-mode-toggle"), cameraControlNote: $("camera-control-note"), cameraControlList: $("camera-control-list"), imageStatistics: $("image-statistics-section"), statFrameBrightness: $("stat-frame-brightness"), statFishBrightness: $("stat-fish-brightness"), statBackgroundBrightness: $("stat-background-brightness"), statContrast: $("stat-contrast"), cameraConditionWarning: $("camera-condition-warning"),
  cameraReset: $("camera-reset-button"), cameraLoadProfile: $("camera-load-profile-button"), cameraSaveProfile: $("camera-save-profile-button"), cameraLock: $("camera-lock-button"), cameraCalibrationMessage: $("camera-calibration-message"),
  cameraProfileConfirmation: $("camera-profile-confirmation"), cameraRecordEmptyReference: $("camera-record-empty-reference-button"), cameraRecordFishReference: $("camera-record-fish-reference-button"), cameraOperatorName: $("camera-operator-name"), cameraInstallationDeviceIdentity: $("camera-install-device-identity"), cameraInstallationHeight: $("camera-install-camera-height"), cameraInstallationAngle: $("camera-install-camera-angle"), cameraInstallationConveyor: $("camera-install-conveyor-position"), cameraInstallationLighting: $("camera-install-lighting-position"), cameraConfirmProfile: $("camera-confirm-profile-button"),
  slider: $("confidence-slider"), settingsThreshold: $("settings-threshold"), confidenceMinus: $("confidence-minus"), confidencePlus: $("confidence-plus"), resetConfidence: $("reset-confidence-button"),
  qualitySlider: $("quality-confidence-slider"), settingsQualityThreshold: $("settings-quality-threshold"), qualityConfidenceMinus: $("quality-confidence-minus"), qualityConfidencePlus: $("quality-confidence-plus"), resetQualityConfidence: $("reset-quality-confidence-button"),
  detectorModelName: $("detector-model-name"), detectorModelStatus: $("detector-model-status"), detectorCheckpoint: $("detector-checkpoint"), detectorDevice: $("detector-device"), detectorThreshold: $("detector-threshold"),
  segmenterModelName: $("segmenter-model-name"), segmenterModelStatus: $("segmenter-model-status"), segmenterCheckpoint: $("segmenter-checkpoint"), segmenterDevice: $("segmenter-device"), segmenterInference: $("segmenter-inference"),
  settingsResolution: $("settings-resolution"), settingsFps: $("settings-fps"), settingsQualityMode: $("settings-quality-mode"), settingsQualityInterval: $("settings-quality-interval"), settingsQualityRoiPadding: $("settings-quality-roi-padding"),
  runtimeDiagnostics: $("runtime-diagnostics"), runtimeDiagnosticsNote: $("runtime-diagnostics-note"), runtimeDiagnosticsValues: $("runtime-diagnostics-values"), debugRuntimeEnabled: $("debug-runtime-enabled"), debugPause: $("debug-pause-button"), hardNegativeList: $("hard-negative-list"),
  debugModel2Raw: $("debug-model2-raw"), debugModel2Associated: $("debug-model2-associated"), debugModel2Rejected: $("debug-model2-rejected"), debugModel2Labels: $("debug-model2-labels"), debugRoiBoundary: $("debug-roi-boundary"),
};

const overlayControls = {
  part_overlays: [$("toggle-part-overlays"), $("settings-part-overlays")], fish_ids: [$("toggle-fish-ids"), $("settings-fish-ids")],
  grades: [$("toggle-grades"), $("settings-grades")], confidence: [$("toggle-confidence"), $("settings-confidence")],
  features: [$("toggle-features"), $("settings-features")], outlines: [$("toggle-outlines"), $("settings-outlines")],
  model2_raw_boxes: [$("debug-model2-raw")], model2_associated_boxes: [$("debug-model2-associated")],
  model2_rejected_boxes: [$("debug-model2-rejected")], model2_labels: [$("debug-model2-labels")], roi_boundary: [$("debug-roi-boundary")],
};
const viewMeta = {
  live: ["Live Inspection", "Real-time Sardinella Lemuru quality inspection"],
  history: ["Inspection History", "Current-session fish inspection records"],
  review: ["Review Queue", "Ungraded fish that need an operator decision"],
  model: ["Model", "Local inference availability and operator-safe configuration"],
};
const theme = getComputedStyle(document.documentElement);
const gradeColors = Object.fromEntries(
  [["Class A", "--success"], ["Class B", "--warning"], ["Class C", "--coral"], ["Rejected", "--danger"], ["Ungraded", "--neutral"]]
    .map(([name, token]) => [name, theme.getPropertyValue(token).trim()])
);

let latestStatus = null;
let streamAttached = false;
let streamReady = false;
let selectedUpload = null;
let uploadInFlight = false;
let selectedView = "live";
let historyState = { page: 1, totalPages: 1 };
let reviewState = { page: 1, totalPages: 1 };
let defaultConfidence = 0.5;
let defaultQualityConfidence = 0.25;
let partQualityChart = null;
let colorTrendChart = null;
let statusRequestInFlight = false;
let cameraCalibrationStatus = null;
let controlRequestInFlight = false;
let controlRevision = 0;
let activePanel = null;
let panelTrigger = null;
let historyRequestVersion = 0;
let reviewRequestVersion = 0;
let historyRequestKey = null;
let reviewRequestKey = null;
let latestEvidenceKey = null;
let recentHistoryKey = null;
let historyRowsKey = null;
let reviewRowsKey = null;
const cameraControlTimers = new Map();
const kpiCards = new Map();
const secondaryKpiCards = new Map();
const featureCards = new Map();
// Match the shipped Model 2 operating point before the first status refresh.
elements.qualitySlider.value = "25";
elements.settingsQualityThreshold.textContent = "25%";

function titleCase(value) { return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function stateClass(value) {
  if (["healthy", "ready", "connected", "running"].includes(value)) return "healthy";
  if (["starting", "connecting", "checking", "attention"].includes(value)) return "warning";
  if (["errored", "error", "unavailable"].includes(value)) return "error";
  return "neutral";
}
function formatTime(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "—" : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function formatPercent(value) { return value == null || Number.isNaN(Number(value)) ? "—" : `${Number(value).toFixed(1)}%`; }
function formatMs(value) { return value == null ? "—" : `${Number(value).toFixed(0)} ms`; }
function setCompactStatus(element, label, value, state) {
  element.className = `compact-status ${stateClass(state)}`;
  element.replaceChildren(document.createElement("i"), Object.assign(document.createElement("b"), { textContent: label }), Object.assign(document.createElement("em"), { textContent: value }));
}
function setText(id, value) { const element = $(id); if (element) element.textContent = value; }
function hasEvents(data) { return Number(data?.counters?.total || 0) > 0; }

function attachStream() {
  if (streamAttached) return;
  elements.video.src = `/api/video-feed?t=${Date.now()}`;
  streamAttached = true;
}
elements.video.addEventListener("load", () => { streamReady = true; elements.placeholder.classList.add("hidden"); });
elements.video.addEventListener("error", () => {
  streamReady = false;
  streamAttached = false;
  elements.video.removeAttribute("src");
  elements.placeholder.classList.remove("hidden");
  elements.placeholderTitle.textContent = "Camera feed interrupted";
  elements.placeholderCopy.textContent = "Reconnecting to the inspection feed…";
});

function renderAlert(data) {
  const modelIssue = data.model_status !== "ready";
  const segmenterIssue = data.runtime_mode !== "part_preview" && ["unavailable", "error"].includes(data.segmenter_status);
  const inspectionIssue = data.inspection_status === "errored";
  const visible = modelIssue || segmenterIssue || inspectionIssue;
  elements.alert.classList.toggle("hidden", !visible);
  if (!visible) return;
  if (modelIssue) {
    elements.alertTitle.textContent = "Detection Model unavailable";
    elements.alertCopy.textContent = data.upload_analysis_available
      ? "Live camera needs the Model 1 fish-detector checkpoint."
      : "Model checkpoint could not be loaded.";
    elements.alertDetails.textContent = data.message || "No diagnostic details supplied.";
  } else if (segmenterIssue) {
    elements.alertTitle.textContent = "Quality Model unavailable";
    elements.alertCopy.textContent = "Fish detection can continue, but fish are recorded as Ungraded until Model 2 is available.";
    elements.alertDetails.textContent = data.segmenter_message || "No diagnostic details supplied.";
  } else {
    elements.alertTitle.textContent = "Inspection unavailable";
    elements.alertCopy.textContent = "The camera or inspection worker needs attention.";
    elements.alertDetails.textContent = data.message || "No diagnostic details supplied.";
  }
}

function renderCamera(data) {
  const inspection = data.inspection_status;
  const running = ["starting", "running"].includes(inspection);
  const preview = data.runtime_mode === "part_preview";
  setCompactStatus($("connection-pill"), "Server", "Online", "ready");
  $("runtime-mode-label").textContent = preview ? "Research mode · Part preview" : "Local inspection · Whole fish";
  $("part-preview-banner").classList.toggle("hidden", !preview);
  elements.shell.dataset.runtimeMode = data.runtime_mode || "whole_fish";
  setCompactStatus(elements.cameraPill, "Camera", titleCase(data.camera_status), data.camera_status);
  const qualityStatus = data.model_info?.quality?.status || data.segmenter_status;
  const modelState = data.model_status !== "ready" ? data.model_status : qualityStatus;
  setCompactStatus(elements.modelPill, preview ? "Part model" : "M1 / M2", preview ? titleCase(data.model_status) : `${titleCase(data.model_status)} / ${titleCase(qualityStatus)}`, preview ? data.model_status : modelState);
  const device = data.model_info?.detector?.device || "—";
  setCompactStatus(elements.devicePill, "Device", String(device).toUpperCase(), device === "cpu" || device === "0" || device === "cuda" ? "healthy" : "neutral");
  elements.inspectionState.className = `inspection-state ${stateClass(inspection)}`;
  elements.inspectionState.replaceChildren(document.createElement("i"), document.createTextNode(titleCase(inspection)));
  elements.liveBadge.className = `live-badge ${stateClass(inspection)}`;
  elements.liveBadge.textContent = inspection === "running" ? "LIVE" : inspection === "starting" ? "STARTING" : "OFFLINE";
  elements.start.disabled = controlRequestInFlight || running || data.model_status !== "ready";
  elements.emptyStart.disabled = elements.start.disabled;
  elements.stop.disabled = controlRequestInFlight || !running;
  elements.reset.disabled = controlRequestInFlight || inspection === "starting";
  elements.analyze.disabled = uploadInFlight || !selectedUpload || !data.upload_analysis_available;
  if (running) attachStream();
  else if (streamAttached) { elements.video.removeAttribute("src"); streamAttached = false; streamReady = false; }
  elements.placeholder.classList.toggle("hidden", inspection === "running" && streamReady);
  if (inspection === "starting") {
    elements.placeholderTitle.textContent = "Starting inspection";
    elements.placeholderCopy.textContent = "Opening the local camera and preparing inference.";
  } else if (inspection === "errored") {
    elements.placeholderTitle.textContent = "Camera unavailable";
    elements.placeholderCopy.textContent = "Check the connected camera, then start inspection again.";
  } else if (inspection === "running" && !streamReady) {
    elements.placeholderTitle.textContent = "Connecting to inspection camera…";
    elements.placeholderCopy.textContent = "Waiting for the first processed camera frame.";
  } else {
    elements.placeholderTitle.textContent = "Camera is currently stopped.";
    elements.placeholderCopy.textContent = data.model_status === "ready"
      ? "Start inspection to connect the camera and view live evidence."
      : "The local model is unavailable. See the status message above.";
  }
  const fps = Number(data.fps || 0);
  elements.feedFps.textContent = `${fps.toFixed(1)} FPS`;
  const frameSize = data.frame_size;
  elements.feedResolution.textContent = Array.isArray(frameSize) ? `${frameSize[0]} × ${frameSize[1]}` : "—";
  elements.settingsResolution.textContent = elements.feedResolution.textContent;
  elements.settingsFps.textContent = `${fps.toFixed(1)} FPS`;
}

function analysisFor(event) { return event?.analysis && typeof event.analysis === "object" ? event.analysis : {}; }

// Evidence-aware verdict helpers. Values in persisted records are fractions; the
// fallback also accepts legacy percentage fields so old sessions remain readable.
const MANUAL_GRADE_CHOICES = ["A", "B", "C", "Rejected"];
function finiteNumber(value) { if (value == null || value === "") return null; const number = Number(value); return Number.isFinite(number) ? number : null; }
function asFraction(value) { const number = finiteNumber(value); return number == null ? null : Math.abs(number) > 1 ? number / 100 : number; }
function scoreText(value) { const fraction = asFraction(value); return fraction == null ? "-" : formatPercent(fraction * 100); }
function eventValue(event, ...keys) {
  const topLevel = event && typeof event === "object" ? event : {}; const analysis = analysisFor(event);
  for (const source of [topLevel, analysis]) for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(source, key) && source[key] !== null && source[key] !== undefined && source[key] !== "") return source[key];
  }
  return undefined;
}
function model1Confidence(event) { const model1 = analysisFor(event).model1_detection; return model1 && typeof model1 === "object" && model1.confidence != null ? model1.confidence : eventValue(event, "model1_detection_confidence", "detection_confidence", "final_confidence", "confidence_percent"); }
function weightedFinalSupport(event) { return eventValue(event, "final_score", "provisional_score", "best_evidence_score", "quality_confidence"); }
function model2BestEvidence(event) { return { grade: eventValue(event, "best_evidence_class", "provisional_grade", "quality"), score: eventValue(event, "best_evidence_score", "provisional_score", "quality_confidence") }; }
function aiFinalGrade(event) { return String(eventValue(event, "ai_final_grade", "final_grade", "quality") || "Ungraded"); }
function manualGrade(event) { return eventValue(event, "manual_grade"); }
function verdictStatus(event) { return String(eventValue(event, "verdict_status") || "NOT_AVAILABLE").toUpperCase(); }
function readableVerdictStatus(value) {
  const labels = { AUTO_GRADED: "Automatically graded", NEEDS_REVIEW: "Needs review", NOT_AVAILABLE: "Verdict status unavailable" };
  return labels[String(value || "").toUpperCase()] || titleCase(value);
}
function verdictReason(event) {
  const stored = eventValue(event, "reason_codes");
  const codes = Array.isArray(stored) ? stored.map(String).filter(Boolean) : typeof stored === "string" ? stored.split(",").map((item) => item.trim()).filter(Boolean) : [];
  const code = String(eventValue(event, "verdict_reason_code") || codes[0] || "");
  if (code && !codes.includes(code)) codes.unshift(code);
  return { code, codes, label: codes.join(", ") || code, text: eventValue(event, "verdict_reason_text") };
}
function needsReview(event) { return verdictStatus(event) === "NEEDS_REVIEW"; }
function observedRegions(event) {
  const stored = eventValue(event, "observed_regions");
  if (Array.isArray(stored)) return stored.map((region) => titleCase(region)).filter(Boolean);
  if (typeof stored === "string" && stored.trim()) return stored.split(",").map((region) => titleCase(region.trim())).filter(Boolean);
  const parts = analysisFor(event).part_results || {};
  return ["Body", "Head", "Tail"].filter((region) => parts[region]?.present === true);
}
function coverageText(event) { return scoreText(eventValue(event, "coverage_percentage", "original_weight_coverage", "coverage")); }
function effectiveWeightText(event) {
  const weights = eventValue(event, "effective_weights");
  if (!weights || typeof weights !== "object" || Array.isArray(weights)) return "-";
  const entries = Object.entries(weights).filter(([, value]) => asFraction(value) != null && asFraction(value) > 0);
  return entries.length ? entries.map(([region, weight]) => `${titleCase(region)} ${scoreText(weight)}`).join("; ") : "-";
}
function partFor(event, region) {
  const parts = analysisFor(event).part_results;
  return parts && typeof parts === "object" ? (parts[region] || parts[region.toLowerCase()] || {}) : {};
}
function partStatus(part) {
  if (part && part.visibility_state) return titleCase(String(part.visibility_state).replaceAll("_", " "));
  if (part && part.status) return titleCase(part.status);
  return part?.present ? "Observed" : "Unknown / not observed";
}
function partContribution(part, grade) {
  const contributions = part?.contributions;
  return contributions && typeof contributions === "object" ? contributions[grade] : undefined;
}
function topStatistics(part) {
  const temporal = part?.temporal_statistics;
  return temporal && typeof temporal === "object" && temporal.top_grade_statistics && typeof temporal.top_grade_statistics === "object" ? temporal.top_grade_statistics : {};
}
function latestFeatureRows(event) {
  return ["Body", "Head", "Tail"].map((region) => {
    const part = partFor(event, region);
    if (!part.present) return [region, partStatus(part), "unmeasured"];
    return [region, `${part.grade || "Unknown"} · ${scoreText(part.grade_confidence)}`, "detected"];
  });
}
function addAnalysisDatum(grid, label, value) {
  const item = document.createElement("span"); const name = document.createElement("small"); const detail = document.createElement("b");
  name.textContent = label; detail.textContent = value == null || value === "" ? "-" : String(value); item.append(name, detail); grid.append(item);
}
function appendAnalysisBlock(body, heading, build) {
  const block = document.createElement("section"); block.className = "analysis-block"; const title = document.createElement("h3"); title.textContent = heading; block.append(title); build(block); body.append(block);
}
function appendFullAnalysis(container, event) {
  if (!event) return;
  const analysis = analysisFor(event); const details = document.createElement("details"); details.className = "full-analysis";
  const summary = document.createElement("summary"); summary.textContent = "Why this verdict and full analysis"; details.append(summary);
  const body = document.createElement("div"); body.className = "full-analysis-body";
  const aiGrade = aiFinalGrade(event); const support = weightedFinalSupport(event); const evidence = model2BestEvidence(event); const manual = manualGrade(event); const reason = verdictReason(event);
  appendAnalysisBlock(body, "Final verdict", (block) => {
    const grid = document.createElement("div"); grid.className = "analysis-data-grid";
    addAnalysisDatum(grid, "Original AI verdict", aiGrade); addAnalysisDatum(grid, "Weighted final support", scoreText(support)); addAnalysisDatum(grid, "Verdict status", readableVerdictStatus(verdictStatus(event))); addAnalysisDatum(grid, "Reason code(s)", reason.label || "-");
    if (manual) addAnalysisDatum(grid, "Manual final grade", manual); block.append(grid);
    if (reason.text) { const text = document.createElement("p"); text.textContent = reason.text; block.append(text); }
  });
  appendAnalysisBlock(body, "Evidence coverage", (block) => {
    const grid = document.createElement("div"); grid.className = "analysis-data-grid";
    const observed = observedRegions(event); addAnalysisDatum(grid, "Observed regions", observed.length ? observed.join(" / ") : "None"); addAnalysisDatum(grid, "Original coverage", coverageText(event)); addAnalysisDatum(grid, "Effective calculation", effectiveWeightText(event)); addAnalysisDatum(grid, "Usable Model 2 frames", eventValue(event, "observation_count") ?? "-"); block.append(grid);
  });
  appendAnalysisBlock(body, "Confidence concepts", (block) => {
    const grid = document.createElement("div"); grid.className = "analysis-data-grid";
    addAnalysisDatum(grid, "Model 1 detection confidence", scoreText(model1Confidence(event))); addAnalysisDatum(grid, "Model 2 best evidence", `${evidence.grade || "-"} ${scoreText(evidence.score)}`); addAnalysisDatum(grid, "Final weighted support", scoreText(support)); addAnalysisDatum(grid, "Best evidence class", evidence.grade || "-"); block.append(grid);
  });
  appendAnalysisBlock(body, "Part analysis", (block) => {
    ["Body", "Head", "Tail"].forEach((region) => {
      const part = partFor(event, region); const stats = topStatistics(part); const contribution = partContribution(part, analysis.provisional_grade || analysis.best_evidence_class);
      const line = document.createElement("p");
      if (!part.present) line.textContent = `${region}: ${partStatus(part)}. This is not a physical missing-part claim.`;
      else line.textContent = `${region}: ${part.grade || "-"} evidence ${scoreText(part.grade_confidence)}; original weight ${scoreText(part.original_weight ?? part.weight)}; effective weight ${scoreText(part.effective_weight ?? part.normalized_weight)}; contribution ${scoreText(contribution)}; observations ${stats.number_of_valid_observations ?? "-"}; mean ${scoreText(stats.mean_confidence)}; max ${scoreText(stats.max_confidence)}; standard deviation ${scoreText(stats.standard_deviation)}.`;
      block.append(line);
    });
  });
  appendAnalysisBlock(body, "All weighted scores", (block) => {
    const scores = analysis.weighted_scores && typeof analysis.weighted_scores === "object" ? analysis.weighted_scores : {};
    const grid = document.createElement("div"); grid.className = "analysis-data-grid";
    ["Class A", "Class B", "Class C", "Rejected"].forEach((grade) => addAnalysisDatum(grid, grade, scoreText(scores[grade]))); block.append(grid);
  });
  const inferenceDebug = analysis.inference_debug;
  if (inferenceDebug && typeof inferenceDebug === "object") appendAnalysisBlock(body, "Inference debug", (block) => {
    const model1 = inferenceDebug.model1_detection || {}; const decision = inferenceDebug.final_decision || {};
    const grid = document.createElement("div"); grid.className = "analysis-data-grid";
    const crop = inferenceDebug.fish_crop_dimensions || {};
    addAnalysisDatum(grid, "Model 1 detection", `Fish #${model1.fish_id ?? event.track_id ?? "-"} ${scoreText(model1.confidence)}`);
    addAnalysisDatum(grid, "Fish crop", crop.width && crop.height ? `${crop.width} x ${crop.height}` : "-");
    addAnalysisDatum(grid, "Model 2 detections", Array.isArray(inferenceDebug.model2_detections) ? inferenceDebug.model2_detections.length : 0);
    addAnalysisDatum(grid, "Final decision", `${decision.grade || "Ungraded"} ${scoreText(decision.support ?? decision.best_evidence_support)}`);
    block.append(grid);
    if (Array.isArray(inferenceDebug.model2_detections) && inferenceDebug.model2_detections.length) {
      const line = document.createElement("p");
      line.textContent = inferenceDebug.model2_detections.map((item) => `${item.region}: ${item.grade} ${scoreText(item.evidence_score)}`).join(" | ");
      block.append(line);
    }
  });
  const stability = analysis.temporal_stability;
  if (stability && typeof stability === "object") appendAnalysisBlock(body, "Temporal stability", (block) => {
    const grid = document.createElement("div"); grid.className = "analysis-data-grid"; addAnalysisDatum(grid, "Ready", stability.ready === true ? "Yes" : stability.ready === false ? "No" : "-"); addAnalysisDatum(grid, "Usable frames", stability.usable_frame_count ?? "-"); addAnalysisDatum(grid, "Candidate frames", stability.candidate_frame_count ?? "-"); addAnalysisDatum(grid, "Winning-grade std. dev.", scoreText(stability.winning_grade_max_standard_deviation)); block.append(grid);
  });
  const bestFrame = analysis.best_frame;
  if (bestFrame && typeof bestFrame === "object") appendAnalysisBlock(body, "Best representative frame", (block) => {
    const grid = document.createElement("div"); grid.className = "analysis-data-grid"; addAnalysisDatum(grid, "Frame", bestFrame.best_frame_id ?? bestFrame.frame_id ?? bestFrame.id ?? "-"); addAnalysisDatum(grid, "Frame score", scoreText(bestFrame.best_frame_score ?? bestFrame.score)); addAnalysisDatum(grid, "Sharpness", bestFrame.frame_quality?.sharpness_score ?? bestFrame.sharpness_score ?? "-"); addAnalysisDatum(grid, "Crop", bestFrame.best_crop_path || "Not saved"); block.append(grid);
  });
  const whole = analysis.color?.["Whole Fish"] || {};
  appendAnalysisBlock(body, "Color metrics", (block) => {
    const text = document.createElement("p"); text.textContent = whole.status === "measured" ? `Whole-fish HSV: hue ${Number(whole.mean_hue_deg).toFixed(1)} degrees; saturation ${scoreText(whole.mean_saturation)}; yellow proxy ${scoreText(whole.yellow_ratio_proxy)}. Classification effect: none; no validated HSV rule is active.` : "HSV measurement unavailable for this fish. Classification effect: none."; block.append(text);
  });
  (analysis.explanation || []).forEach((text) => { const line = document.createElement("p"); line.textContent = String(text); body.append(line); });
  details.append(body); container.append(details);
}
function renderLatest(data) {
  const evidenceKey = JSON.stringify([data.current_detection, data.latest_event, data.runtime_mode]);
  if (latestEvidenceKey === evidenceKey) return;
  const expanded = Boolean(elements.latestFeatures.querySelector("details")?.open);
  latestEvidenceKey = evidenceKey;
  const event = data.current_detection || data.latest_event; const hasEvent = Boolean(event); const liveTrack = Boolean(event?.live_track); const status = verdictStatus(event); const manual = manualGrade(event); const aiGrade = aiFinalGrade(event); const support = weightedFinalSupport(event); const evidence = model2BestEvidence(event); const reason = verdictReason(event);
  elements.resultEmpty.classList.toggle("hidden", hasEvent); elements.resultActive.classList.toggle("hidden", !hasEvent); elements.latestVerdictDetails.classList.toggle("hidden", !hasEvent);
  const verdictClass = aiGrade === "Rejected" ? "error" : aiGrade === "Ungraded" || needsReview(event) ? "warning" : "healthy";
  elements.decision.className = `decision ${hasEvent ? verdictClass : "neutral"}`; elements.decision.textContent = hasEvent ? (liveTrack ? "LIVE FISH" : readableVerdictStatus(status).toUpperCase()) : "WAITING";
  elements.latestFeatures.replaceChildren(); latestFeatureRows(event).forEach(([label, value, className]) => { const row = document.createElement("div"); row.className = "feature-row"; const name = document.createElement("span"); name.textContent = label; const valueElement = document.createElement("b"); valueElement.textContent = value; valueElement.classList.add(className); row.append(name, valueElement); elements.latestFeatures.append(row); });
  appendFullAnalysis(elements.latestFeatures, event);
  if (expanded) elements.latestFeatures.querySelector("details")?.setAttribute("open", "");
  $("result-panel").style.setProperty("--grade-color", gradeColors[manual || aiGrade] || gradeColors.Ungraded);
  if (data.runtime_mode === "part_preview") {
    elements.resultEmpty.classList.remove("hidden");
    elements.resultActive.classList.add("hidden");
    elements.latestVerdictDetails.classList.add("hidden");
    elements.resultEmptyTitle.textContent = "Part evidence preview";
    elements.resultEmptyCopy.textContent = "Raw part detections do not establish a whole-fish grade.";
    elements.decision.textContent = "RESEARCH";
    elements.decision.className = "decision warning";
    elements.latestFeatures.replaceChildren();
    elements.classificationPolicy.textContent = "Fish counting and final grading are disabled in this mode.";
    return;
  }
  elements.resultEmptyTitle.textContent = "Ready for the next fish";
  elements.resultEmptyCopy.textContent = "The current track and its quality evidence will appear here.";
  if (!hasEvent) { elements.classificationPolicy.textContent = "Model 1 detection, Model 2 evidence, and final weighted support are reported separately."; return; }
  elements.resultFish.textContent = event.fish_label || `Fish #${event.track_id}`; elements.resultLabel.textContent = manual ? `Manual: ${manual}` : aiGrade; const verdictLabel = elements.resultLabel.closest(".grade-readout")?.querySelector("span"); if (verdictLabel) verdictLabel.textContent = liveTrack ? "LIVE VERDICT" : "FINAL VERDICT";
  elements.confidence.textContent = scoreText(support); elements.confidenceBar.style.width = `${Math.max(0, Math.min(100, (asFraction(support) || 0) * 100))}%`;
  const confidenceLabel = elements.confidence.closest("div")?.querySelector("dt"); if (confidenceLabel) confidenceLabel.textContent = "Weighted final support";
  elements.resultProcessing.textContent = formatMs(event.processing_time_ms); elements.resultTime.textContent = liveTrack ? "Live track" : formatTime(event.timestamp);
  elements.latestVerdictStatus.textContent = manual ? "Manual final grade recorded" : readableVerdictStatus(status); elements.latestVerdictReason.textContent = reason.text || (needsReview(event) ? "The AI did not meet the configured final-verdict acceptance rules." : "No structured verdict reason was supplied.");
  elements.latestDetectionConfidence.textContent = scoreText(model1Confidence(event)); elements.latestModel2Evidence.textContent = `${evidence.grade || "-"} ${scoreText(evidence.score)}`; elements.latestCoverage.textContent = `${coverageText(event)} (${observedRegions(event).join(" / ") || "no regions"})`; elements.latestEffectiveWeights.textContent = effectiveWeightText(event);
  elements.classificationPolicy.textContent = manual ? `Manual final grade: ${manual}. Original AI verdict remains ${aiGrade}; Model 1 detection and Model 2 evidence are retained separately.` : needsReview(event) ? "This is not a Rejected fish. The AI does not have enough reliable evidence for an automatic final grade." : "Model 1 detection, Model 2 evidence, and final weighted support are separate measurements.";
}

function kpiCard(label, value, note, emphasis = false, cache = kpiCards) {
  let record = cache.get(label);
  if (!record) {
    const card = document.createElement("article");
    const labelEl = document.createElement("span"); labelEl.className = "kpi-label"; labelEl.append(document.createElement("i"), document.createTextNode(label));
    const valueEl = document.createElement("strong");
    const noteEl = document.createElement("small");
    card.append(labelEl, valueEl, noteEl);
    record = { card, valueEl, noteEl };
    cache.set(label, record);
  }
  record.card.className = `kpi-card${emphasis ? " emphasis" : ""}`;
  record.card.style.setProperty("--grade-color", gradeColors[label] || theme.getPropertyValue("--ocean-700").trim());
  const nextValue = String(value);
  if (record.valueEl.textContent !== nextValue) record.valueEl.textContent = nextValue;
  if (record.noteEl.textContent !== note) record.noteEl.textContent = note;
  return record.card;
}

function reconcileCards(container, cache, cards, makeCard) {
  const wanted = new Set(cards.map(([name]) => name));
  for (const [name, record] of cache) {
    if (!wanted.has(name)) {
      record.card.remove();
      cache.delete(name);
    }
  }
  cards.forEach((cardData, index) => {
    const card = makeCard(cardData);
    if (container.children[index] !== card) container.insertBefore(card, container.children[index] || null);
  });
}
function renderKpis(data) {
  const counters = data.counters || {}; const grades = data.quality_counters || {}; const total = Number(counters.total || 0);
  const percentage = (name) => total ? `${((Number(grades[name] || 0) / total) * 100).toFixed(1)}% of inspected fish` : "No fish this session";
  const accepted = Number(grades["Class A"] || 0) + Number(grades["Class B"] || 0) + Number(grades["Class C"] || 0);
  const rejected = Number(grades.Rejected || 0); const live = data.current_detection || data.latest_event;
  const liveGrade = live ? (aiFinalGrade(live) || "Ungraded") : "-";
  const averageSupport = data.analytics?.average_final_support ?? data.analytics?.average_best_evidence_support;
  const cards = [["Total fish", total, "Current session", true]];
  Object.entries(grades).forEach(([name, count]) => cards.push([name, count || 0, percentage(name)]));
  const secondary = [
    ["Acceptance rate", total ? `${((accepted / total) * 100).toFixed(1)}%` : "-", "Grade A, B, or C final verdicts"],
    ["Rejection rate", total ? `${((rejected / total) * 100).toFixed(1)}%` : "-", "Real Rejected-class verdicts"],
    ["Active fish", data.active_fish_count || 0, "Live tracks"],
    ["Inspection FPS", `${Number(data.fps || 0).toFixed(1)}`, "Frames per second"],
    ["Current fish ID", live ? `Fish #${live.track_id}` : "-", live?.live_track ? "Current Model 1 track" : "No active track"],
    ["Current final grade", liveGrade, live?.live_track ? "Current Model 2 evidence" : "Latest completed fish"],
    ["Average support", averageSupport == null ? "-" : formatPercent(averageSupport), "Average weighted support"]
  ];
  elements.kpiGrid.querySelector(".metric-waiting")?.remove();
  if (data.runtime_mode === "part_preview") {
    cards.splice(0, cards.length, ["Counting disabled", "—", "Research part preview"], ["Grading disabled", "—", "No whole-fish verdicts"]);
    secondary.splice(0, secondary.length, ["Inspection FPS", `${Number(data.fps || 0).toFixed(1)}`, "Preview frames per second"]);
  }
  reconcileCards(elements.kpiGrid, kpiCards, cards, ([label, value, note, emphasis]) => kpiCard(label, value, note, Boolean(emphasis)));
  reconcileCards($("secondary-kpi-grid"), secondaryKpiCards, secondary, ([label, value, note]) => kpiCard(label, value, note, false, secondaryKpiCards));
  elements.threshold.textContent = formatPercent(Number(data.detection_confidence_threshold) * 100);
}

function renderFeatureCounters(data) {
  const counters = Object.entries(data.feature_counters || {});
  reconcileCards(elements.featureCounterGrid, featureCards, counters, ([name, detail]) => {
    let record = featureCards.get(name);
    if (!record) {
      const card = document.createElement("article");
      const title = document.createElement("span"); title.textContent = name;
      const value = document.createElement("strong");
      const note = document.createElement("small");
      card.append(title, value, note);
      record = { card, value, note };
      featureCards.set(name, record);
    }
    record.card.className = `feature-counter${detail.available ? "" : " unavailable"}`;
    const valueText = detail.available ? String(detail.count || 0) : "Not measured";
    if (record.value.textContent !== valueText) record.value.textContent = valueText;
    const noteText = String(detail.source || "");
    if (record.note.textContent !== noteText) record.note.textContent = noteText;
    return record.card;
  });
}

function ensureAnalysisCharts() {
  if (partQualityChart && colorTrendChart) return;
  const grid = document.querySelector("#analytics-section .analytics-grid"); if (!grid) return;
  const makeCard = (title) => { const card = document.createElement("article"); card.className = "chart-card surface"; const heading = document.createElement("h3"); heading.textContent = title; const chart = document.createElement("div"); chart.className = "bar-chart"; card.append(heading, chart); grid.append(card); return chart; };
  partQualityChart = makeCard("Average part quality scores");
  colorTrendChart = makeCard("Colour measurement trend");
}
function appendMetricRow(container, label, fraction, value) {
  const row = document.createElement("div"); row.className = "bar-row"; const name = document.createElement("span"); name.textContent = label; const track = document.createElement("i"); track.style.setProperty("--bar-width", `${Math.max(0, Math.min(100, fraction * 100))}%`); const amount = document.createElement("b"); amount.textContent = value; row.append(name, track, amount); container.append(row);
}
function renderAnalytics(analytics) {
  ensureAnalysisCharts();
  const hasData = Boolean(analytics?.has_data);
  [elements.gradeChartEmpty, elements.featureChartEmpty, elements.throughputChartEmpty, elements.confidenceChartEmpty].forEach((element) => element.classList.toggle("hidden", hasData));
  elements.gradeChart.classList.toggle("hidden", !hasData); elements.featureChart.classList.toggle("hidden", !hasData); elements.throughputChart.classList.toggle("hidden", !hasData);
  const confidenceAvailable = analytics?.average_grade_confidence != null;
  elements.confidenceChart.classList.toggle("hidden", !confidenceAvailable);
  elements.confidenceChartEmpty.classList.toggle("hidden", confidenceAvailable);
  if (!hasData) {
    [partQualityChart, colorTrendChart].filter(Boolean).forEach((chart) => {
      chart.classList.add("chart-empty");
      chart.textContent = "Waiting for inspection data";
    });
    return;
  }
  [partQualityChart, colorTrendChart].filter(Boolean).forEach((chart) => chart.classList.remove("chart-empty"));
  const distribution = analytics.grade_distribution || {}; const total = Object.values(distribution).reduce((sum, value) => sum + Number(value || 0), 0);
  let offset = 0; const segments = []; const colors = { ...gradeColors };
  Object.keys(distribution).forEach((name) => { if (!colors[name]) colors[name] = gradeColors.Ungraded; });
  Object.entries(colors).filter(([name]) => name in distribution).forEach(([name, color]) => { const portion = total ? (Number(distribution[name] || 0) / total) * 100 : 0; segments.push(`${color} ${offset}% ${offset + portion}%`); offset += portion; });
  elements.gradeDonut.style.background = `conic-gradient(${segments.join(", ")})`;
  elements.gradeTotal.textContent = String(total);
  elements.gradeLegend.replaceChildren();
  Object.entries(colors).filter(([name]) => name in distribution).forEach(([name, color]) => { const row = document.createElement("div"); row.className = "legend-row"; row.style.setProperty("--legend-color", color); const label = document.createElement("span"); label.textContent = name; const value = document.createElement("b"); value.textContent = String(distribution[name] || 0); row.append(label, value); elements.gradeLegend.append(row); });
  elements.featureChart.replaceChildren();
  const features = analytics.feature_counts || {}; const measurable = Object.entries(features).filter(([, detail]) => detail.available);
  const max = Math.max(1, ...measurable.map(([, detail]) => Number(detail.count || 0)));
  measurable.forEach(([name, detail]) => { const row = document.createElement("div"); row.className = "bar-row"; const label = document.createElement("span"); label.textContent = name; const track = document.createElement("i"); track.style.setProperty("--bar-width", `${(Number(detail.count || 0) / max) * 100}%`); const value = document.createElement("b"); value.textContent = String(detail.count || 0); row.append(label, track, value); elements.featureChart.append(row); });
  if (partQualityChart) { partQualityChart.replaceChildren(); const scores = analytics.average_part_quality_scores || {}; ["Body", "Head", "Tail"].forEach((region) => { const score = scores[region]; appendMetricRow(partQualityChart, region, Number(score || 0) / 100, score == null ? "—" : formatPercent(score)); }); }
  if (colorTrendChart) { colorTrendChart.replaceChildren(); const trend = analytics.color_trend || []; if (!trend.length) { const note = document.createElement("span"); note.textContent = "No measured HSV samples"; colorTrendChart.append(note); } else trend.slice(-6).forEach((sample) => appendMetricRow(colorTrendChart, formatTime(sample.timestamp), Number(sample.mean_hue_deg || 0) / 360, `${Number(sample.mean_hue_deg).toFixed(1)}° / ${formatPercent(sample.yellow_ratio_proxy)}`)); }
  const throughput = analytics.throughput || []; const maxTotal = Math.max(1, ...throughput.map((item) => Number(item.total || 0))); const denominator = Math.max(1, throughput.length - 1);
  const points = throughput.map((item, index) => `${(index / denominator) * 560},${166 - (Number(item.total || 0) / maxTotal) * 142}`).join(" ");
  elements.throughputLine.setAttribute("points", points || "0,166");
  elements.throughputCaption.textContent = `${total} fish · ${throughput.length} observations`;
  if (confidenceAvailable) elements.averageGradeConfidence.textContent = `M1 raw ${formatPercent(analytics.average_detection_confidence)} · final support ${formatPercent(analytics.average_grade_confidence)}`;
}

function detectedFeatures(event) {
  const regions = [...new Set((event?.parts || []).map((part) => part.region).filter(Boolean))];
  return regions.length ? regions.join(", ") : "No Model 2 regions";
}
function historyCell(value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value == null || value === "" ? "—" : String(value);
  if (className) cell.className = className;
  if (className === "grade-cell") cell.style.setProperty("--grade-color", gradeColors[value] || gradeColors.Ungraded);
  return cell;
}
function fishHistoryCell(event) {
  const cell = document.createElement("td");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "history-open";
  button.textContent = event.fish_label || `Fish #${event.track_id}`;
  button.setAttribute("aria-label", `Open full analysis for ${button.textContent}`);
  button.addEventListener("click", () => openHistoryAnalysis(event));
  cell.append(button);
  return cell;
}
function labelHistoryRows(body) {
  const table = body.closest("table");
  table.setAttribute("role", "table");
  table.setAttribute("aria-label", body === elements.historyBody ? "Inspection history" : body === elements.reviewBody ? "Review queue" : "Recent inspections");
  const headings = [...table.querySelectorAll("thead th")].map((header) => header.textContent.trim());
  table.querySelectorAll("thead, tbody").forEach((group) => group.setAttribute("role", "rowgroup"));
  table.querySelectorAll("tr").forEach((row) => row.setAttribute("role", "row"));
  body.querySelectorAll("tr").forEach((row) => {
    [...row.children].forEach((cell, index) => {
      cell.dataset.label = headings[index] || "Detail";
      cell.setAttribute("role", "cell");
    });
  });
}
function renderRecent(history) {
  const rows = (history || []).slice(0, 10);
  const key = JSON.stringify(rows);
  if (key === recentHistoryKey) return;
  recentHistoryKey = key;
  elements.recentEmpty.classList.toggle("hidden", rows.length > 0); elements.recentWrap.classList.toggle("hidden", rows.length === 0); elements.recentHistoryBody.replaceChildren();
  rows.forEach((event) => {
    const row = document.createElement("tr"); row.append(fishHistoryCell(event), historyCell(formatTime(event.timestamp)), historyCell(aiFinalGrade(event), "grade-cell"), historyCell(scoreText(weightedFinalSupport(event))), historyCell(detectedFeatures(event), "status-note"), historyCell(formatMs(event.processing_time_ms))); elements.recentHistoryBody.append(row);
  });
  labelHistoryRows(elements.recentHistoryBody);
}

function manualChoiceForDisplay(value) { return String(value || "").replace(/^Class\s+/i, ""); }
function appendManualReviewPanel(container, event) {
  const currentManual = manualGrade(event); if (!needsReview(event) && !currentManual) return;
  const panel = document.createElement("section"); panel.className = "manual-review-panel";
  const heading = document.createElement("h3"); heading.textContent = "Manual review"; const original = document.createElement("p");
  original.textContent = `Original AI verdict: ${aiFinalGrade(event)}. Weighted final support: ${scoreText(weightedFinalSupport(event))}. This evidence is retained when a manual grade is recorded.`;
  const controls = document.createElement("div"); controls.className = "manual-review-controls"; const label = document.createElement("label"); label.textContent = "Manual final grade";
  const select = document.createElement("select"); select.setAttribute("aria-label", "Manual final grade"); const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Select an operator grade"; select.append(empty);
  MANUAL_GRADE_CHOICES.forEach((grade) => { const option = document.createElement("option"); option.value = grade; option.textContent = grade === "Rejected" ? "Rejected" : `Class ${grade}`; option.selected = manualChoiceForDisplay(currentManual) === grade; select.append(option); });
  label.append(select); const save = document.createElement("button"); save.type = "button"; save.className = "button primary"; save.textContent = currentManual ? "Update manual grade" : "Record manual grade"; save.disabled = !select.value;
  const message = document.createElement("p"); message.className = "manual-review-message"; if (currentManual) message.textContent = `Manual final grade currently recorded as ${currentManual}.`;
  select.addEventListener("change", () => { save.disabled = !select.value; message.textContent = ""; message.classList.remove("error"); });
  save.addEventListener("click", () => submitManualReview(event, select.value, save, message)); controls.append(label, save); panel.append(heading, original, controls, message); container.append(panel);
}
async function submitManualReview(event, selectedGrade, button, message) {
  if (!selectedGrade || event?.track_id == null) return;
  button.disabled = true; message.classList.remove("error"); message.textContent = "Saving manual final grade...";
  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(String(event.track_id))}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ manual_grade: selectedGrade }) });
    const payload = await response.json().catch(() => ({})); if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    const updated = payload.item && typeof payload.item === "object" ? payload.item : null;
    message.textContent = "Manual final grade saved. The original AI verdict is unchanged.";
    if (updated) openHistoryAnalysis(updated); await getStatus(); await loadHistory(); await loadReviewQueue();
  } catch (error) { message.classList.add("error"); message.textContent = error instanceof Error ? error.message : "Manual review could not be saved."; button.disabled = false; }
}
function openHistoryAnalysis(event) {
  let dialog = $("history-analysis-dialog");
  if (!dialog) { dialog = document.createElement("dialog"); dialog.id = "history-analysis-dialog"; dialog.className = "history-analysis-dialog"; document.body.append(dialog); }
  dialog.replaceChildren(); const close = document.createElement("button"); close.className = "icon-button"; close.type = "button"; close.textContent = "x"; close.setAttribute("aria-label", "Close analysis"); close.addEventListener("click", () => dialog.close());
  const title = document.createElement("h2"); title.textContent = `${event.fish_label || `Fish #${event.track_id}`} full analysis`; const content = document.createElement("div"); appendFullAnalysis(content, event); content.querySelector("details")?.setAttribute("open", ""); appendManualReviewPanel(content, event); dialog.append(close, title, content);
  title.id = "history-analysis-title";
  dialog.setAttribute("aria-labelledby", title.id);
  if (typeof dialog.showModal === "function") dialog.showModal(); else dialog.setAttribute("open", "");
}

function renderModel(data) {
  const preview = data.runtime_mode === "part_preview";
  const intro = $("view-model").querySelector(".view-intro");
  intro.querySelector("h2").textContent = preview ? "Research part-model status" : "Local two-model status";
  intro.querySelector("p:last-child").textContent = preview
    ? "The preview model shows raw part evidence. Whole-fish tracking, counting, and final grading are disabled."
    : "Model 1 locates and tracks fish. Model 2 supplies head, body, and tail evidence for grading.";
  elements.detectorModelName.closest(".model-card").querySelector(".eyebrow").textContent = preview ? "RESEARCH · PART MODEL" : "MODEL 1 · FISH DETECTION";
  const detector = data.model_info?.detector || {}; const segmenter = data.model_info?.quality || {}; const diagnostic = segmenter.diagnostics || {};
  elements.detectorModelName.textContent = detector.name || "Fish detector"; elements.detectorModelStatus.textContent = titleCase(detector.status); elements.detectorCheckpoint.textContent = detector.checkpoint_resolved ? "Resolved" : "Unresolved"; elements.detectorDevice.textContent = detector.device || "—"; elements.detectorThreshold.textContent = formatPercent(Number(detector.confidence_threshold) * 100);
  elements.segmenterModelName.textContent = segmenter.name || "Quality Model"; elements.segmenterModelStatus.textContent = titleCase(segmenter.status); elements.segmenterCheckpoint.textContent = segmenter.checkpoint_resolved ? "Resolved" : "Unresolved"; elements.segmenterDevice.textContent = segmenter.device || "—"; elements.segmenterInference.textContent = diagnostic.last_inference_ms == null ? "No measurement" : `${diagnostic.last_inference_ms} ms`;
}

function diagnosticsRow(label, value) {
  const row = document.createElement("div"); const term = document.createElement("dt"); const detail = document.createElement("dd");
  term.textContent = label; detail.textContent = value == null || value === "" ? "—" : String(value); row.append(term, detail); return row;
}
function renderRuntimeDiagnostics(data) {
  const diagnostics = data.runtime_diagnostics || {}; const performance = diagnostics.performance || {}; const timings = performance.timings || {}; const queue = diagnostics.frame_queue || {};
  const enabled = Boolean(diagnostics.enabled); elements.debugRuntimeEnabled.checked = enabled; elements.debugPause.disabled = !enabled; elements.debugPause.textContent = diagnostics.paused ? "Resume inspection" : "Pause on current frame";
  elements.runtimeDiagnosticsNote.textContent = enabled
    ? "Debug mode is bounded and research-only. Raw candidates and saved crops do not change production predictions, grading, or history."
    : "Off by default. Enable only while investigating Model 1 / Model 2 runtime behavior.";
  elements.runtimeDiagnosticsValues.replaceChildren(
    diagnosticsRow("Capture FPS", performance.calls_per_second?.camera_capture == null ? "—" : `${Number(performance.calls_per_second.camera_capture).toFixed(1)}`),
    diagnosticsRow("Processing FPS", performance.calls_per_second?.processed_frame == null ? "—" : `${Number(performance.calls_per_second.processed_frame).toFixed(1)}`),
    diagnosticsRow("Dropped stale frames", queue.dropped_frames ?? "—"), diagnosticsRow("Queue depth", queue.depth ?? "—"),
    diagnosticsRow("Model 1 p50 / p95", timings.model1_inference?.p50 == null ? "—" : `${timings.model1_inference.p50} / ${timings.model1_inference.p95} ms`),
    diagnosticsRow("Model 2 p50 / p95", timings.model2_inference?.p50 == null ? "—" : `${timings.model2_inference.p50} / ${timings.model2_inference.p95} ms`),
    diagnosticsRow("JPEG p50 / p95", timings.jpeg_encoding?.p50 == null ? "—" : `${timings.jpeg_encoding.p50} / ${timings.jpeg_encoding.p95} ms`),
    diagnosticsRow("Frontend render p50 / p95", timings.frontend_render?.p50 == null ? "Waiting for browser telemetry" : `${timings.frontend_render.p50} / ${timings.frontend_render.p95} ms`),
    diagnosticsRow("Model 2 raw / associated / rejected / unassigned", `${diagnostics.model2_raw_candidates ?? 0} / ${diagnostics.model2_associated_parts ?? 0} / ${diagnostics.model2_rejected_candidates ?? 0} / ${diagnostics.model2_unassigned_candidates ?? 0}`),
  );
  elements.hardNegativeList.replaceChildren(); const records = Array.isArray(diagnostics.hard_negative_records) ? diagnostics.hard_negative_records : [];
  if (!records.length) { elements.hardNegativeList.textContent = enabled ? "No bounded candidates saved yet." : "Enable diagnostics to create bounded research-only candidates."; return; }
  records.slice(-8).reverse().forEach((record) => {
    const row = document.createElement("div"); row.className = "verdict-summary-row";
    const text = document.createElement("span"); text.textContent = `Frame ${record.frame_number} · Fish #${record.track_id} · detector ${formatPercent(Number(record.confidence) * 100)} · Model 2 compatible: ${record.model2_compatible_part_evidence ? "yes" : "no"}`;
    row.append(text);
    if (record.operator_label === "NOT_FISH") row.append(Object.assign(document.createElement("strong"), { textContent: "NOT_FISH" }));
    else { const mark = document.createElement("button"); mark.className = "text-button"; mark.type = "button"; mark.textContent = "Mark NOT_FISH"; mark.addEventListener("click", () => markHardNegative(record.capture_id)); row.append(mark); }
    elements.hardNegativeList.append(row);
  });
}
async function postRuntimeDiagnostics(payload) {
  try { const response = await fetch("/api/debug/runtime", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`); render(data); }
  catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Runtime diagnostics unavailable"; elements.alertCopy.textContent = "The debug action was not applied."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown error."; }
}
async function markHardNegative(captureId) {
  if (!captureId) return;
  try { const response = await fetch(`/api/debug/hard-negatives/${encodeURIComponent(String(captureId))}/not-fish`, { method: "POST" }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`); await getStatus(); }
  catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "NOT_FISH label not saved"; elements.alertCopy.textContent = "The production prediction was not changed."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown error."; }
}

function renderGradingSettings(data) {
  let section = $("grading-settings-section");
  if (!section) {
    section = document.createElement("section");
    section.id = "grading-settings-section";
    section.className = "settings-section";
    section.innerHTML = "<h3>Grading policy</h3><p class=\"muted\">Read-only policy snapshot. Weighted support is not a calibrated probability.</p><dl class=\"settings-list\"><div><dt>Mode</dt><dd id=\"settings-grading-mode\">—</dd></div><div><dt>Part evidence threshold</dt><dd id=\"settings-part-evidence-threshold\">—</dd></div><div><dt>Final verdict threshold</dt><dd id=\"settings-final-verdict-threshold\">—</dd></div><div><dt>Minimum grade margin</dt><dd id=\"settings-grade-margin\">—</dd></div><div><dt>Required coverage</dt><dd id=\"settings-coverage\">—</dd></div><div><dt>Fixed weights</dt><dd id=\"settings-part-weights\">—</dd></div></dl>";
    const cameraSection = [...elements.settingsDrawer.querySelectorAll(".settings-section")]
      .find((candidate) => candidate.querySelector("h3")?.textContent?.trim() === "Camera");
    if (cameraSection) cameraSection.before(section); else elements.settingsDrawer.append(section);
  }
  const rules = data.grading_rules && typeof data.grading_rules === "object" ? data.grading_rules : {};
  const text = (id, value) => { const target = $(id); if (target) target.textContent = value; };
  const fraction = (value) => Number.isFinite(Number(value)) ? formatPercent(Number(value) * 100) : "—";
  const weights = rules.part_weights && typeof rules.part_weights === "object" ? rules.part_weights : {};
  text("settings-grading-mode", rules.grading_mode || "No production policy");
  text("settings-part-evidence-threshold", fraction(rules.minimum_part_confidence));
  text("settings-final-verdict-threshold", fraction(rules.final_verdict_threshold));
  text("settings-grade-margin", fraction(rules.minimum_grade_margin));
  text("settings-coverage", fraction(rules.minimum_original_weight_coverage));
  text("settings-part-weights", `Body ${fraction(weights.Body)} / Head ${fraction(weights.Head)} / Tail ${fraction(weights.Tail)}`);
}

function syncSettings(data) {
  const threshold = Math.round(Number(data.detection_confidence_threshold || 0) * 100);
  elements.slider.value = String(threshold); elements.settingsThreshold.textContent = `${threshold}%`;
  const qualityThreshold = Math.round(Number(data.quality_confidence_threshold || 0) * 100);
  elements.qualitySlider.value = String(qualityThreshold); elements.settingsQualityThreshold.textContent = `${qualityThreshold}%`;
  const qualityRuntime = data.model_info?.quality || {};
  elements.settingsQualityMode.textContent = qualityRuntime.quality_inference_mode || qualityRuntime.diagnostics?.quality_inference_mode || "—";
  const interval = Number(qualityRuntime.quality_interval);
  elements.settingsQualityInterval.textContent = Number.isFinite(interval) && interval > 0 ? `Every ${interval} processed frame${interval === 1 ? "" : "s"}` : "—";
  const padding = Number(qualityRuntime.roi_padding ?? qualityRuntime.diagnostics?.roi_padding);
  elements.settingsQualityRoiPadding.textContent = Number.isFinite(padding) ? `${padding} px` : "—";
  const policyLocked = Boolean(data.policy_settings_locked ?? ["starting", "running"].includes(data.inspection_status));
  [elements.slider, elements.confidenceMinus, elements.confidencePlus, elements.resetConfidence, elements.qualitySlider, elements.qualityConfidenceMinus, elements.qualityConfidencePlus, elements.resetQualityConfidence]
    .forEach((control) => { if (control) { control.disabled = policyLocked; control.title = policyLocked ? "Stop inspection before changing this policy setting." : ""; } });
  const display = data.display_settings || {};
  Object.entries(overlayControls).forEach(([name, controls]) => controls.forEach((control) => { if (control) control.checked = Boolean(display[name]); }));
  renderGradingSettings(data);
}

function render(data) {
  const renderStarted = performance.now(); latestStatus = data; defaultConfidence = Number(data.default_confidence_threshold ?? defaultConfidence); defaultQualityConfidence = Number(data.default_quality_confidence_threshold ?? defaultQualityConfidence); renderAlert(data); renderCamera(data); renderLatest(data); renderKpis(data); renderFeatureCounters(data); renderAnalytics(data.analytics); renderRecent(data.recent_history); renderModel(data); syncSettings(data); renderRuntimeDiagnostics(data);
  if (data.runtime_diagnostics?.enabled) fetch("/api/debug/frontend-metrics", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ render_ms: performance.now() - renderStarted }) }).catch(() => {});
  const reviewCount = Number(data.review_queue_count || 0);
  $("review-count").textContent = String(reviewCount);
  $("review-count").classList.toggle("hidden", reviewCount === 0);
  if (data.runtime_mode === "part_preview") {
    elements.recentEmpty.textContent = "Fish counting is disabled in part preview.";
    $("analytics-disclosure").classList.add("hidden");
    $("analytics-section").classList.add("hidden");
    elements.featureCounterGrid.closest(".feature-section").classList.add("hidden");
  } else {
    elements.recentEmpty.textContent = "No completed inspections yet. Fish appear here after crossing the counting line.";
    $("analytics-disclosure").classList.remove("hidden");
    $("analytics-section").classList.remove("hidden");
    elements.featureCounterGrid.closest(".feature-section").classList.remove("hidden");
  }
  if (selectedView === "history") loadHistory();
  else if (selectedView === "review") loadReviewQueue();
}

function renderBackendUnavailable() {
  elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Connection lost"; elements.alertCopy.textContent = "Cannot reach the local backend. Displayed results may be out of date."; elements.alertDetails.textContent = "Check that the local server is running. The dashboard will reconnect automatically.";
  setCompactStatus($("connection-pill"), "Server", "Offline", "error");
  setCompactStatus(elements.cameraPill, "Camera", "Unknown", "neutral");
  setCompactStatus(elements.modelPill, "Models", "Unknown", "neutral");
  elements.liveBadge.textContent = "CONNECTION LOST";
  elements.liveBadge.className = "live-badge error";
  elements.video.removeAttribute("src"); streamAttached = false; streamReady = false;
  elements.placeholder.classList.remove("hidden");
  elements.placeholderTitle.textContent = "Waiting for the local server";
  elements.placeholderCopy.textContent = "The feed will reconnect when the backend is available.";
  [elements.start, elements.emptyStart, elements.stop, elements.reset, elements.analyze].forEach((button) => { button.disabled = true; });
}
async function getStatus() {
  // A slow response must never overtake the next poll and redraw stale UI.
  if (statusRequestInFlight || controlRequestInFlight) return;
  statusRequestInFlight = true;
  const revision = controlRevision;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (revision === controlRevision && !controlRequestInFlight) render(data);
  } catch (_) {
    if (revision === controlRevision && !controlRequestInFlight) renderBackendUnavailable();
  } finally {
    statusRequestInFlight = false;
  }
}
async function postControl(path) {
  if (controlRequestInFlight) return;
  controlRequestInFlight = true;
  controlRevision += 1;
  const feedback = $("control-feedback");
  const activity = path.endsWith("/start") ? "Starting inspection…" : path.endsWith("/stop") ? "Stopping inspection…" : "Resetting session…";
  feedback.textContent = activity;
  feedback.classList.remove("error");
  elements.inspectionState.setAttribute("aria-busy", "true");
  [elements.start, elements.emptyStart, elements.stop, elements.reset].forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(path, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    controlRequestInFlight = false;
    render(data);
    feedback.textContent = data.inspection_status === "errored" ? (data.message || "Inspection needs attention.") : path.endsWith("/reset") ? "New session ready. Previous records remain in local history." : path.endsWith("/stop") ? (data.inspection_status === "stopped" ? "Inspection stopped. Camera released." : "Stop requested. Waiting for the camera to be released.") : "Inspection requested. Waiting for the camera.";
    feedback.classList.toggle("error", data.inspection_status === "errored");
  } catch (error) {
    feedback.textContent = error instanceof Error ? error.message : "Inspection request failed.";
    feedback.classList.add("error");
    controlRequestInFlight = false;
    await getStatus();
  } finally {
    controlRequestInFlight = false;
    elements.inspectionState.removeAttribute("aria-busy");
  }
}
async function postSettings(payload) {
  try { const response = await fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`); render(data); }
  catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Settings not applied"; elements.alertCopy.textContent = "The active configuration could not be updated."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown settings error."; }
}

function showView(view) {
  if (!viewMeta[view]) return;
  selectedView = view;
  Object.keys(viewMeta).forEach((name) => $("view-" + name).classList.toggle("hidden", name !== view));
  const [title, subtitle] = viewMeta[view]; elements.pageTitle.textContent = title; elements.pageSubtitle.textContent = subtitle;
  document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (view === "history") loadHistory(true);
  if (view === "review") loadReviewQueue(true);
  window.scrollTo({ top: 0, behavior: motionPreference() });
}
function motionPreference() { return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"; }
function updateScrim() {
  const anyOpen = elements.settingsDrawer.classList.contains("open") || elements.cameraCalibrationDrawer.classList.contains("open") || !elements.exportModal.classList.contains("hidden");
  elements.scrim.classList.toggle("hidden", !anyOpen);
  elements.shell.inert = anyOpen;
  document.body.classList.toggle("panel-open", anyOpen);
}
function openPanel(panel) {
  if (!activePanel) panelTrigger = document.activeElement;
  [elements.settingsDrawer, elements.cameraCalibrationDrawer, elements.exportModal].forEach((candidate) => {
    const selected = candidate === panel;
    candidate.inert = !selected;
    candidate.setAttribute("aria-hidden", String(!selected));
    if (candidate === elements.exportModal) candidate.classList.toggle("hidden", !selected);
    else candidate.classList.toggle("open", selected);
  });
  activePanel = panel;
  updateScrim();
  (panel.querySelector("button:not(:disabled)") || panel).focus();
}
function closePanel(panel) {
  if (panel !== activePanel) return;
  panel.classList.remove("open");
  if (panel === elements.exportModal) panel.classList.add("hidden");
  panel.inert = true;
  panel.setAttribute("aria-hidden", "true");
  activePanel = null;
  updateScrim();
  panelTrigger?.focus();
}
function openSettings() { openPanel(elements.settingsDrawer); }
function closeSettings() { closePanel(elements.settingsDrawer); }
function openExport() { openPanel(elements.exportModal); }
function closeExport() { closePanel(elements.exportModal); }
function openCameraCalibration() { openPanel(elements.cameraCalibrationDrawer); loadCameraCalibration(); }
function closeCameraCalibration() { closePanel(elements.cameraCalibrationDrawer); }
document.addEventListener("keydown", (event) => {
  if (!activePanel) return;
  if (event.key === "Escape") { event.preventDefault(); closePanel(activePanel); return; }
  if (event.key !== "Tab") return;
  const focusable = [...activePanel.querySelectorAll('button, input, select, summary, a[href], [tabindex="0"]')]
    .filter((control) => !control.disabled && control.getClientRects().length && !control.closest("[inert]"));
  const first = focusable[0], last = focusable.at(-1);
  if (!first) { event.preventDefault(); activePanel.focus(); }
  else if (event.shiftKey && (document.activeElement === first || !activePanel.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && (document.activeElement === last || !activePanel.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
});

function cameraNumber(value) { return value == null || Number.isNaN(Number(value)) ? "-" : String(Number(value)); }
function setCameraCalibrationMessage(message, error = false) { elements.cameraCalibrationMessage.textContent = message; elements.cameraCalibrationMessage.classList.toggle("error", error); }
function cameraEditable(status) { return Boolean(status?.available && status?.calibration_mode && !status?.locked); }
function fillCameraField(input, value) { if (input && !input.value.trim() && value != null) input.value = String(value); }
function researchInstallationPayload() {
  return {
    device_identity: elements.cameraInstallationDeviceIdentity.value.trim(),
    camera_height: elements.cameraInstallationHeight.value.trim(),
    camera_angle: elements.cameraInstallationAngle.value.trim(),
    conveyor_position: elements.cameraInstallationConveyor.value.trim(),
    lighting_position: elements.cameraInstallationLighting.value.trim(),
  };
}

function renderCameraControls(status) {
  const properties = status?.capabilities?.properties || {};
  const actual = status?.actual || {};
  const editable = cameraEditable(status);
  elements.cameraControlList.replaceChildren();
  if (!Object.keys(properties).length) {
    const note = document.createElement("p"); note.className = "muted";
    note.textContent = "Camera controls will appear when the connected device reports its capabilities.";
    elements.cameraControlList.append(note);
  }
  Object.entries(properties).forEach(([key, capability]) => {
    const supported = Boolean(capability?.supported);
    const row = document.createElement("div"); row.className = `camera-control${supported ? "" : " unsupported"}`;
    const head = document.createElement("div"); head.className = "camera-control-head";
    const label = document.createElement("label"); label.textContent = capability?.label || titleCase(key);
    const output = document.createElement("output"); output.textContent = supported ? (capability?.control === "toggle" ? (actual[key] ? "ON" : "OFF") : cameraNumber(actual[key])) : "Not supported";
    head.append(label, output); row.append(head);
    if (!supported) {
      const note = document.createElement("small"); note.textContent = capability?.reason || "Not supported by this camera/backend"; row.append(note);
    } else if (capability?.control === "toggle") {
      const control = document.createElement("label"); control.className = "camera-control-toggle"; control.append(document.createTextNode("Physical camera control"));
      const input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(actual[key]); input.disabled = !editable; input.setAttribute("aria-label", capability?.label || key);
      input.id = `camera-control-${key}`; label.htmlFor = input.id;
      input.addEventListener("change", () => postCameraSettings({ [key]: input.checked })); control.append(input); row.append(control);
    } else {
      const range = Array.isArray(capability?.range) && capability.range.length === 2 ? capability.range : null;
      const input = document.createElement("input"); input.type = range ? "range" : "number"; input.value = actual[key] == null ? "" : String(actual[key]); input.disabled = !editable; input.setAttribute("aria-label", capability?.label || key);
      input.id = `camera-control-${key}`; label.htmlFor = input.id; output.htmlFor = input.id;
      if (range) { input.min = String(range[0]); input.max = String(range[1]); input.step = capability?.step == null ? "any" : String(capability.step); }
      else { input.step = "any"; input.placeholder = "Driver value"; }
      const apply = () => { if (input.value !== "") postCameraSettings({ [key]: Number(input.value) }); };
      if (range) input.addEventListener("input", () => { output.textContent = input.value; clearTimeout(cameraControlTimers.get(key)); cameraControlTimers.set(key, setTimeout(apply, 160)); });
      input.addEventListener("change", () => { clearTimeout(cameraControlTimers.get(key)); apply(); }); row.append(input);
      const note = document.createElement("small"); note.textContent = range ? "Hardware range reported by the active backend." : "This backend does not report a safe slider range; enter a value from the camera driver."; row.append(note);
    }
    elements.cameraControlList.append(row);
  });
}

function renderCameraCalibration(status) {
  cameraCalibrationStatus = status;
  const camera = status?.camera || {};
  const available = Boolean(status?.available);
  const locked = Boolean(status?.locked);
  const calibration = Boolean(status?.calibration_mode);
  elements.cameraDeviceName.textContent = camera.device_name || "Camera unavailable";
  elements.cameraConnection.textContent = available ? "Connected to the shared inspection capture" : (status?.capabilities?.reason || "Start inspection to connect");
  elements.cameraLockBadge.textContent = locked ? "LOCKED" : "UNLOCKED"; elements.cameraLockBadge.classList.toggle("locked", locked);
  elements.cameraCaptureResolution.textContent = Array.isArray(camera.capture_resolution) ? `${camera.capture_resolution[0]} x ${camera.capture_resolution[1]}` : "-";
  elements.cameraProcessingResolution.textContent = Array.isArray(latestStatus?.frame_size) ? `${latestStatus.frame_size[0]} x ${latestStatus.frame_size[1]}` : "-";
  elements.cameraHardwareFps.textContent = cameraNumber(camera.fps); elements.cameraBackend.textContent = camera.backend || "-";
  elements.cameraProfileName.textContent = camera.profile_name || "Not saved"; elements.cameraLockState.textContent = locked ? "Locked" : "Unlocked";
  elements.calibrationMode.checked = calibration; elements.calibrationMode.disabled = !available || locked;
  elements.cameraControlNote.textContent = available ? (locked ? "Unlock to calibrate" : calibration ? "Validated hardware controls" : "Enable Calibration Mode to adjust") : "Camera is not running";
  elements.imageStatistics.classList.toggle("hidden", !calibration);
  const stats = status?.image_statistics || {};
  elements.statFrameBrightness.textContent = cameraNumber(stats.frame_brightness); elements.statFishBrightness.textContent = cameraNumber(stats.fish_roi_brightness); elements.statBackgroundBrightness.textContent = cameraNumber(stats.background_brightness); elements.statContrast.textContent = cameraNumber(stats.fish_background_contrast);
  const warning = status?.condition_warning; elements.cameraConditionWarning.classList.toggle("hidden", !warning); elements.cameraConditionWarning.textContent = warning?.message || "";
  const editable = cameraEditable(status); elements.cameraReset.disabled = !editable; elements.cameraLoadProfile.disabled = !editable; elements.cameraSaveProfile.disabled = !editable; elements.cameraLock.disabled = !available;
  elements.cameraLock.textContent = locked ? "Unlock camera settings" : "Lock inspection camera";
  const profile = status?.profile || {};
  const confirmation = profile?.operator_confirmation || {};
  const references = status?.pending_reference_scenes || profile?.reference_scenes || {};
  const emptyReference = references?.empty_conveyor;
  elements.cameraProfileConfirmation.textContent = confirmation?.confirmed
    ? `Confirmed${confirmation.confirmed_at ? ` · ${confirmation.confirmed_at}` : ""}`
    : (emptyReference ? "Empty-conveyor reference recorded; save then confirm" : "Record an empty-conveyor reference first");
  const installation = profile?.installation || {};
  fillCameraField(elements.cameraOperatorName, confirmation?.operator_name);
  fillCameraField(elements.cameraInstallationDeviceIdentity, installation?.device_identity);
  fillCameraField(elements.cameraInstallationHeight, installation?.camera_height);
  fillCameraField(elements.cameraInstallationAngle, installation?.camera_angle);
  fillCameraField(elements.cameraInstallationConveyor, installation?.conveyor_position);
  fillCameraField(elements.cameraInstallationLighting, installation?.lighting_position);
  elements.cameraRecordEmptyReference.disabled = !editable;
  elements.cameraRecordFishReference.disabled = !editable;
  elements.cameraConfirmProfile.disabled = !editable;
  renderCameraControls(status);
}

async function loadCameraCalibration() {
  try {
    const response = await fetch("/api/camera/settings"); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Camera settings unavailable.");
    renderCameraCalibration(data);
  } catch (error) { setCameraCalibrationMessage(error instanceof Error ? error.message : "Camera settings unavailable.", true); }
}
async function cameraRequest(path, payload = undefined, successMessage = null) {
  try {
    const options = payload === undefined ? { method: "POST" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    const response = await fetch(path, options); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Camera request was not accepted.");
    renderCameraCalibration(data.camera || data); setCameraCalibrationMessage(data.success === false ? "The camera returned a different value; review the readback." : successMessage || (data.policy_session_id ? "Camera setting applied. Active evidence was reset and a new policy session started." : "Camera setting applied."), data.success === false); return data;
  } catch (error) { setCameraCalibrationMessage(error instanceof Error ? error.message : "Camera request failed.", true); await loadCameraCalibration(); return null; }
}
function postCameraSettings(values) { return cameraRequest("/api/camera/settings", values); }


function ensureHistoryVerdictFilter() {
  const filters = elements.historyFilter?.parentElement; if (!filters) return;
  if (!$("history-verdict-status")) {
    const label = document.createElement("label"); label.textContent = "Verdict status"; const select = document.createElement("select"); select.id = "history-verdict-status";
    [["", "All verdicts"], ["NEEDS_REVIEW", "Needs review"]].forEach(([value, text]) => { const option = document.createElement("option"); option.value = value; option.textContent = text; select.append(option); });
    label.append(select); filters.insertBefore(label, elements.historyFilter);
  }
  elements.historyVerdict = $("history-verdict-status");
  const header = elements.historyBody?.closest("table")?.querySelector("thead tr");
  if (header && !header.dataset.verdictColumns) {
    header.dataset.verdictColumns = "true"; header.replaceChildren(); ["Fish ID", "Timestamp", "Original AI verdict", "Weighted support", "Model 1 detection", "Model 2 best evidence", "Coverage", "Verdict status", "Reason", "Manual final", "Processing"].forEach((name) => { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = name; header.append(cell); });
  }
}
function renderHistoryRows(rows) {
  const key = JSON.stringify(rows); if (historyRowsKey === key) return; historyRowsKey = key;
  elements.historyBody.replaceChildren();
  rows.forEach((event) => {
    const evidence = model2BestEvidence(event); const reason = verdictReason(event); const row = document.createElement("tr"); const fish = fishHistoryCell(event);
    row.append(fish, historyCell(formatTime(event.timestamp)), historyCell(aiFinalGrade(event), "grade-cell"), historyCell(scoreText(weightedFinalSupport(event))), historyCell(scoreText(model1Confidence(event))), historyCell(`${evidence.grade || "-"} ${scoreText(evidence.score)}`, "status-note"), historyCell(coverageText(event)), historyCell(readableVerdictStatus(verdictStatus(event)), needsReview(event) ? "review-status" : "status-note"), historyCell(reason.text || reason.label || "-", "status-note"), historyCell(manualGrade(event) || "-", manualGrade(event) ? "manual-grade" : "status-note"), historyCell(formatMs(event.processing_time_ms)));
    elements.historyBody.append(row);
  });
  labelHistoryRows(elements.historyBody);
}
function reviewFilterMatches(event) {
  const search = elements.historySearch?.value.trim().replace(/fish|#/gi, "").trim(); if (search && !String(event.track_id ?? "").includes(search)) return false;
  const grade = elements.historyGrade?.value; if (grade && aiFinalGrade(event).toLowerCase() !== grade.toLowerCase()) return false;
  const feature = elements.historyFeature?.value; if (feature && !observedRegions(event).some((region) => region.toLowerCase() === feature.toLowerCase())) return false;
  const minimum = finiteNumber(elements.historyConfidence?.value); return minimum == null || (asFraction(weightedFinalSupport(event)) || 0) * 100 >= minimum;
}
function setHistoryPageState(state, total, itemCount) {
  state.totalPages = Math.max(1, Math.ceil(total / 25)); state.page = Math.min(Math.max(1, state.page), state.totalPages); elements.historyPage.textContent = `Page ${state.page} of ${state.totalPages}`; elements.historyPrev.disabled = state.page <= 1; elements.historyNext.disabled = state.page >= state.totalPages; return itemCount;
}
async function loadHistory(resetPage = false) {
  ensureHistoryVerdictFilter(); if (resetPage) historyState.page = 1;
  const requestKey = JSON.stringify([historyState.page, elements.historySearch.value, elements.historyGrade.value, elements.historyFeature.value, elements.historyConfidence.value, elements.historyVerdict?.value, latestStatus?.durable_session?.session_id]);
  if (historyRequestKey === requestKey) return;
  historyRequestKey = requestKey;
  const version = ++historyRequestVersion;
  elements.historyEmpty.textContent = "No inspection records match these filters.";
  try {
    if (elements.historyVerdict?.value === "NEEDS_REVIEW") {
      const response = await fetch("/api/reviews?include_reviewed=true"); const data = await response.json(); if (version !== historyRequestVersion) return; if (!response.ok) throw new Error(data.detail || "Review filter unavailable");
      const filtered = (data.items || []).filter(reviewFilterMatches); const total = filtered.length; setHistoryPageState(historyState, total, total); const pageItems = filtered.slice((historyState.page - 1) * 25, historyState.page * 25);
      elements.historyEmpty.classList.toggle("hidden", pageItems.length > 0); elements.historyWrap.classList.toggle("hidden", pageItems.length === 0); renderHistoryRows(pageItems); return;
    }
    const query = new URLSearchParams({ page: String(historyState.page), page_size: "25" }); if (elements.historySearch.value.trim()) query.set("search", elements.historySearch.value.trim()); if (elements.historyGrade.value) query.set("grade", elements.historyGrade.value); if (elements.historyFeature.value) query.set("feature", elements.historyFeature.value); if (elements.historyConfidence.value) query.set("min_confidence", elements.historyConfidence.value);
    const response = await fetch(`/api/history?${query}`); const data = await response.json(); if (version !== historyRequestVersion) return; if (!response.ok) throw new Error(data.detail || "History unavailable"); historyState.totalPages = data.total_pages || 1; historyState.page = data.page || 1; elements.historyPage.textContent = `Page ${historyState.page} of ${historyState.totalPages}`; elements.historyPrev.disabled = historyState.page <= 1; elements.historyNext.disabled = historyState.page >= historyState.totalPages;
    const rows = data.items || []; elements.historyEmpty.classList.toggle("hidden", rows.length > 0); elements.historyWrap.classList.toggle("hidden", rows.length === 0); renderHistoryRows(rows);
  } catch (error) { if (version === historyRequestVersion) { elements.historyEmpty.textContent = error instanceof Error ? error.message : "History unavailable."; elements.historyEmpty.classList.remove("hidden"); elements.historyWrap.classList.add("hidden"); } }
  finally { if (version === historyRequestVersion) historyRequestKey = null; }
}
function renderReviewRows(rows) {
  const key = JSON.stringify(rows); if (reviewRowsKey === key) return; reviewRowsKey = key;
  elements.reviewBody.replaceChildren();
  rows.forEach((event) => {
    const reason = verdictReason(event); const row = document.createElement("tr"); const fish = fishHistoryCell(event);
    const action = document.createElement("button"); action.type = "button"; action.className = "button quiet review-action"; action.textContent = "Review"; action.addEventListener("click", () => openHistoryAnalysis(event)); const actionCell = document.createElement("td"); actionCell.append(action);
    row.append(fish, historyCell(formatTime(event.timestamp)), historyCell(aiFinalGrade(event), "grade-cell"), historyCell(scoreText(weightedFinalSupport(event))), historyCell(coverageText(event)), historyCell(reason.text || reason.label || "-", "status-note"), historyCell(manualGrade(event) || "-", manualGrade(event) ? "manual-grade" : "status-note"), actionCell); elements.reviewBody.append(row);
  });
  labelHistoryRows(elements.reviewBody);
}
async function loadReviewQueue(resetPage = false) {
  if (!elements.reviewBody) return; if (resetPage) reviewState.page = 1;
  const requestKey = JSON.stringify([reviewState.page, latestStatus?.durable_session?.session_id]);
  if (reviewRequestKey === requestKey) return;
  reviewRequestKey = requestKey;
  const version = ++reviewRequestVersion;
  elements.reviewEmpty.textContent = "No fish currently need review.";
  try {
    const response = await fetch("/api/reviews"); const data = await response.json(); if (version !== reviewRequestVersion) return; if (!response.ok) throw new Error(data.detail || "Review queue unavailable"); const allItems = Array.isArray(data.items) ? data.items : []; const total = finiteNumber(data.total) ?? allItems.length;
    reviewState.totalPages = Math.max(1, Math.ceil(total / 25)); reviewState.page = Math.min(Math.max(1, reviewState.page), reviewState.totalPages); elements.reviewPage.textContent = `Page ${reviewState.page} of ${reviewState.totalPages}`; elements.reviewPrev.disabled = reviewState.page <= 1; elements.reviewNext.disabled = reviewState.page >= reviewState.totalPages;
    const rows = allItems.slice((reviewState.page - 1) * 25, reviewState.page * 25); elements.reviewEmpty.classList.toggle("hidden", rows.length > 0); elements.reviewWrap.classList.toggle("hidden", rows.length === 0); renderReviewRows(rows);
  } catch (error) { if (version === reviewRequestVersion) { elements.reviewEmpty.textContent = error instanceof Error ? error.message : "Review queue unavailable."; elements.reviewEmpty.classList.remove("hidden"); elements.reviewWrap.classList.add("hidden"); } }
  finally { if (version === reviewRequestVersion) reviewRequestKey = null; }
}

function setConfidence(value, apply = false) {
  const normalized = Math.max(0, Math.min(100, Math.round(Number(value)))); elements.slider.value = String(normalized); elements.settingsThreshold.textContent = `${normalized}%`;
  if (apply) postSettings({ detection_confidence_threshold: normalized / 100 });
}
function setQualityConfidence(value, apply = false) {
  const normalized = Math.max(0, Math.min(100, Math.round(Number(value)))); elements.qualitySlider.value = String(normalized); elements.settingsQualityThreshold.textContent = `${normalized}%`;
  if (apply) postSettings({ quality_confidence_threshold: normalized / 100 });
}
function setOverlay(name, checked) { overlayControls[name].forEach((control) => { control.checked = checked; }); postSettings({ display_settings: { [name]: checked } }); }
function toggleUpload(show) { elements.uploadWorkspace.classList.toggle("hidden", !show); if (show) elements.uploadWorkspace.scrollIntoView({ behavior: motionPreference(), block: "start" }); }

document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll(".nav-item[data-anchor]").forEach((button) => button.addEventListener("click", () => { showView("live"); $("analytics-disclosure").open = true; setTimeout(() => { $(button.dataset.anchor).scrollIntoView({ behavior: motionPreference(), block: "start" }); document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === button)); }, 30); }));
elements.sidebarToggle.addEventListener("click", () => {
  const collapsed = elements.shell.classList.toggle("sidebar-collapsed");
  elements.sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  elements.sidebarToggle.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
});
elements.start.addEventListener("click", () => postControl("/api/inspection/start")); elements.emptyStart.addEventListener("click", () => postControl("/api/inspection/start")); elements.stop.addEventListener("click", () => postControl("/api/inspection/stop"));
elements.reset.addEventListener("click", () => { if (!hasEvents(latestStatus) || window.confirm("Reset this session? Live counters will reset. Saved inspection records will remain in the local database.")) postControl("/api/session/reset"); });
elements.uploadMode.addEventListener("click", () => toggleUpload(true)); elements.closeUpload.addEventListener("click", () => toggleUpload(false));
elements.uploadInput.addEventListener("change", () => { selectedUpload = elements.uploadInput.files?.[0] || null; elements.uploadResults.classList.add("hidden"); if (!selectedUpload) { elements.uploadStatus.textContent = "No image selected"; elements.uploadImage.classList.add("hidden"); elements.analyze.disabled = true; return; } elements.uploadStatus.textContent = "Ready"; elements.uploadNote.textContent = `${selectedUpload.name} is ready for separate local analysis.`; const reader = new FileReader(); reader.onload = () => { elements.uploadImage.src = reader.result; elements.uploadImage.classList.remove("hidden"); }; reader.readAsDataURL(selectedUpload); elements.analyze.disabled = !latestStatus?.upload_analysis_available; });
elements.analyze.addEventListener("click", async () => { if (!selectedUpload || uploadInFlight) return; uploadInFlight = true; elements.analyze.disabled = true; elements.uploadStatus.textContent = "Analyzing"; elements.uploadNote.textContent = latestStatus?.runtime_mode === "part_preview" ? "Running the research part model. No fish grades or counts are produced." : "Running Model 1, then Model 2 on each detected fish crop."; try { const response = await fetch("/api/analyze-image", { method: "POST", headers: { "Content-Type": selectedUpload.type || "application/octet-stream", "X-Filename": selectedUpload.name }, body: selectedUpload }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`); const isPartPreview = payload.analysis_mode === "part_preview"; elements.uploadImage.src = payload.annotated_image; elements.uploadFishCount.textContent = String(isPartPreview ? payload.parts_detected : payload.fish_detected); elements.uploadCountLabel.textContent = isPartPreview ? "Parts detected" : "Fish found"; renderUploadQuality(payload.quality_counters); elements.uploadResults.classList.remove("hidden"); elements.uploadStatus.textContent = "Complete"; elements.uploadNote.textContent = isPartPreview ? "Raw part-preview test complete. It does not create fish counts or live inspection records." : payload.model2_available ? "Model 1 fish detections and Model 2 part/grade evidence are shown." : "Detection complete; the Quality Model is unavailable."; } catch (error) { elements.uploadStatus.textContent = "Error"; elements.uploadNote.textContent = error instanceof Error ? error.message : "Image analysis failed."; } finally { uploadInFlight = false; elements.analyze.disabled = !selectedUpload || !latestStatus?.upload_analysis_available; } });
function renderUploadQuality(counters) { const grid = $("upload-quality-counter-grid"); grid.replaceChildren(); Object.entries(counters || {}).forEach(([name, value]) => { const card = document.createElement("article"); card.className = "feature-counter"; card.append(Object.assign(document.createElement("span"), { textContent: name }), Object.assign(document.createElement("strong"), { textContent: String(value) }), Object.assign(document.createElement("small"), { textContent: "Image analysis only" })); grid.append(card); }); }

ensureHistoryVerdictFilter();
elements.viewAll.addEventListener("click", () => showView("history")); elements.historyFilter.addEventListener("click", () => loadHistory(true)); elements.historyPrev.addEventListener("click", () => { if (historyState.page > 1) { historyState.page -= 1; loadHistory(); } }); elements.historyNext.addEventListener("click", () => { if (historyState.page < historyState.totalPages) { historyState.page += 1; loadHistory(); } });
elements.reviewPrev.addEventListener("click", () => { if (reviewState.page > 1) { reviewState.page -= 1; loadReviewQueue(); } }); elements.reviewNext.addEventListener("click", () => { if (reviewState.page < reviewState.totalPages) { reviewState.page += 1; loadReviewQueue(); } });
elements.settingsButton.addEventListener("click", openSettings); elements.sidebarSettings.addEventListener("click", openSettings); elements.closeSettings.addEventListener("click", closeSettings); elements.openConfidence.addEventListener("click", openSettings); elements.cameraSettingsButton.addEventListener("click", openCameraCalibration); elements.closeCameraCalibration.addEventListener("click", closeCameraCalibration); elements.scrim.addEventListener("click", () => { closeSettings(); closeCameraCalibration(); closeExport(); });
elements.exportButton.addEventListener("click", openExport); elements.sidebarExport.addEventListener("click", openExport); elements.closeExport.addEventListener("click", closeExport); elements.cancelExport.addEventListener("click", closeExport);
elements.calibrationMode.addEventListener("change", () => postCameraSettings({ calibration_mode: elements.calibrationMode.checked }));
  elements.cameraReset.addEventListener("click", () => cameraRequest("/api/camera/reset")); elements.cameraLoadProfile.addEventListener("click", () => cameraRequest("/api/camera/profile/load")); elements.cameraSaveProfile.addEventListener("click", () => cameraRequest("/api/camera/profile/save", undefined, "Inspection profile saved. Record references and explicitly confirm when the physical setup is reviewed.")); elements.cameraLock.addEventListener("click", () => cameraRequest("/api/camera/lock", { locked: !Boolean(cameraCalibrationStatus?.locked) }));
  elements.cameraRecordEmptyReference.addEventListener("click", () => cameraRequest("/api/camera/reference", { scene_type: "empty_conveyor" }, "Empty-conveyor reference recorded. Save the profile before confirmation."));
  elements.cameraRecordFishReference.addEventListener("click", () => cameraRequest("/api/camera/reference", { scene_type: "representative_fish" }, "Representative-fish reference recorded for acquisition monitoring only."));
  elements.cameraConfirmProfile.addEventListener("click", () => {
    const installation = researchInstallationPayload();
    const missing = Object.entries(installation).filter(([, value]) => !value).map(([key]) => key.replaceAll("_", " "));
    if (missing.length) { setCameraCalibrationMessage(`Confirmation needs: ${missing.join(", ")}.`, true); return; }
    cameraRequest("/api/camera/profile/confirm", { operator_confirmed: true, operator_name: elements.cameraOperatorName.value.trim() || null, installation }, "Calibration profile explicitly confirmed. Lock the camera before collecting study data.");
  });
elements.slider.addEventListener("input", () => setConfidence(elements.slider.value)); elements.slider.addEventListener("change", () => setConfidence(elements.slider.value, true)); elements.confidenceMinus.addEventListener("click", () => setConfidence(Number(elements.slider.value) - 1, true)); elements.confidencePlus.addEventListener("click", () => setConfidence(Number(elements.slider.value) + 1, true)); elements.resetConfidence.addEventListener("click", () => setConfidence(defaultConfidence * 100, true));
elements.qualitySlider.addEventListener("input", () => setQualityConfidence(elements.qualitySlider.value)); elements.qualitySlider.addEventListener("change", () => setQualityConfidence(elements.qualitySlider.value, true)); elements.qualityConfidenceMinus.addEventListener("click", () => setQualityConfidence(Number(elements.qualitySlider.value) - 1, true)); elements.qualityConfidencePlus.addEventListener("click", () => setQualityConfidence(Number(elements.qualitySlider.value) + 1, true)); elements.resetQualityConfidence.addEventListener("click", () => setQualityConfidence(defaultQualityConfidence * 100, true));
elements.debugRuntimeEnabled.addEventListener("change", () => postRuntimeDiagnostics({ enabled: elements.debugRuntimeEnabled.checked }));
elements.debugPause.addEventListener("click", () => postRuntimeDiagnostics({ paused: !Boolean(latestStatus?.runtime_diagnostics?.paused) }));
Object.entries(overlayControls).forEach(([name, controls]) => controls.filter(Boolean).forEach((control) => control.addEventListener("change", () => setOverlay(name, control.checked))));
document.querySelectorAll("input[name=export-range]").forEach((input) => input.addEventListener("change", () => { elements.dateRange.classList.toggle("hidden", document.querySelector("input[name=export-range]:checked").value !== "custom"); }));
elements.exportForm.addEventListener("submit", async (event) => { event.preventDefault(); const button = $("confirm-export-button"); const fields = [...document.querySelectorAll(".export-fields input:checked")].map((input) => input.value); if (!fields.length) return; const payload = { format: document.querySelector("input[name=export-format]:checked").value, range: document.querySelector("input[name=export-range]:checked").value, fields, complete_analysis: true, start_date: elements.exportStartDate.value || null, end_date: elements.exportEndDate.value || null }; button.disabled = true; try { const response = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); if (!response.ok) { const error = await response.json(); throw new Error(error.detail || "Export failed."); } const blob = await response.blob(); const anchor = document.createElement("a"); const disposition = response.headers.get("content-disposition") || ""; const name = disposition.match(/filename="?([^";]+)"?/)?.[1] || `sardinella_inspection.${payload.format}`; anchor.href = URL.createObjectURL(blob); anchor.download = name; document.body.append(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(anchor.href); closeExport(); } catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Export unavailable"; elements.alertCopy.textContent = "Inspection data could not be exported."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown export error."; } finally { button.disabled = false; } });

getStatus(); setInterval(getStatus, 900);
