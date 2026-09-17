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
  uploadMode: $("upload-mode-button"), closeUpload: $("close-upload-button"), uploadWorkspace: $("upload-workspace"), uploadInput: $("upload-input"), analyze: $("analyze-button"), uploadStatus: $("upload-status"), uploadNote: $("upload-note"), uploadImage: $("upload-result-image"), uploadResults: $("upload-results"), uploadFishCount: $("upload-fish-count"), uploadCountLabel: $("upload-count-label"),
  kpiGrid: $("kpi-grid"), featureCounterGrid: $("feature-counter-grid"), threshold: $("threshold"), openConfidence: $("open-confidence-button"),
  gradeChartEmpty: $("grade-chart-empty"), gradeChart: $("grade-chart"), gradeDonut: $("grade-donut"), gradeTotal: $("grade-total"), gradeLegend: $("grade-legend"),
  featureChartEmpty: $("feature-chart-empty"), featureChart: $("feature-chart"), throughputChartEmpty: $("throughput-chart-empty"), throughputChart: $("throughput-chart"), throughputLine: $("throughput-line"), throughputCaption: $("throughput-caption"),
  confidenceChartEmpty: $("confidence-chart-empty"), confidenceChart: $("confidence-chart"), averageGradeConfidence: $("average-grade-confidence"),
  recentEmpty: $("recent-empty"), recentWrap: $("recent-table-wrap"), recentHistoryBody: $("recent-history-body"), viewAll: $("view-all-button"),
  historyEmpty: $("history-empty"), historyWrap: $("history-table-wrap"), historyBody: $("history-body"), historySearch: $("history-search"), historyGrade: $("history-grade"), historyFeature: $("history-feature"), historyConfidence: $("history-confidence"), historyFilter: $("history-filter-button"), historyPrev: $("history-prev"), historyNext: $("history-next"), historyPage: $("history-page"),
  exportButton: $("export-button"), sidebarExport: $("sidebar-export"), exportModal: $("export-modal"), exportForm: $("export-form"), closeExport: $("close-export-button"), cancelExport: $("cancel-export-button"), dateRange: $("date-range"), exportStartDate: $("export-start-date"), exportEndDate: $("export-end-date"),
  settingsButton: $("settings-button"), sidebarSettings: $("sidebar-settings"), settingsDrawer: $("settings-drawer"), closeSettings: $("close-settings-button"), scrim: $("scrim"),
  slider: $("confidence-slider"), settingsThreshold: $("settings-threshold"), confidenceMinus: $("confidence-minus"), confidencePlus: $("confidence-plus"), resetConfidence: $("reset-confidence-button"),
  qualitySlider: $("quality-confidence-slider"), settingsQualityThreshold: $("settings-quality-threshold"), qualityConfidenceMinus: $("quality-confidence-minus"), qualityConfidencePlus: $("quality-confidence-plus"), resetQualityConfidence: $("reset-quality-confidence-button"),
  detectorModelName: $("detector-model-name"), detectorModelStatus: $("detector-model-status"), detectorCheckpoint: $("detector-checkpoint"), detectorDevice: $("detector-device"), detectorThreshold: $("detector-threshold"),
  segmenterModelName: $("segmenter-model-name"), segmenterModelStatus: $("segmenter-model-status"), segmenterCheckpoint: $("segmenter-checkpoint"), segmenterDevice: $("segmenter-device"), segmenterInference: $("segmenter-inference"),
  settingsResolution: $("settings-resolution"), settingsFps: $("settings-fps"),
};

const overlayControls = {
  masks: [$("toggle-masks"), $("settings-masks")], fish_ids: [$("toggle-fish-ids"), $("settings-fish-ids")],
  grades: [$("toggle-grades"), $("settings-grades")], confidence: [$("toggle-confidence"), $("settings-confidence")],
  features: [$("toggle-features"), $("settings-features")], outlines: [$("toggle-outlines"), $("settings-outlines")],
};
const viewMeta = {
  live: ["Live Inspection", "Real-time Sardinella Lemuru quality inspection"],
  history: ["Inspection History", "Current-session fish inspection records"],
  model: ["Model", "Local inference availability and operator-safe configuration"],
};
const gradeColors = { "Class A": "#49a977", "Class B": "#4388e5", "Class C": "#9464c9", Rejected: "#d55c67", Ungraded: "#a8b5c6" };

let latestStatus = null;
let streamAttached = false;
let streamReady = false;
let selectedUpload = null;
let uploadInFlight = false;
let selectedView = "live";
let historyState = { page: 1, totalPages: 1 };
let defaultConfidence = 0.5;
let defaultQualityConfidence = 0.25;
let partQualityChart = null;
let colorTrendChart = null;
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
elements.video.addEventListener("error", () => { streamReady = false; });

function renderAlert(data) {
  const modelIssue = data.model_status !== "ready";
  const segmenterIssue = ["unavailable", "error"].includes(data.segmenter_status);
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
  setCompactStatus(elements.cameraPill, "Camera", titleCase(data.camera_status), data.camera_status);
  const qualityStatus = data.model_info?.quality?.status || data.segmenter_status;
  setCompactStatus(elements.modelPill, "M1 / M2", `${data.model_status === "ready" ? "Ready" : titleCase(data.model_status)} / ${titleCase(qualityStatus)}`, data.model_status === "ready" && qualityStatus === "ready" ? "ready" : qualityStatus);
  const device = data.model_info?.detector?.device || "—";
  setCompactStatus(elements.devicePill, "Device", String(device).toUpperCase(), device === "cpu" || device === "0" || device === "cuda" ? "healthy" : "neutral");
  elements.inspectionState.className = `inspection-state ${stateClass(inspection)}`;
  elements.inspectionState.replaceChildren(document.createElement("i"), document.createTextNode(titleCase(inspection)));
  elements.liveBadge.className = `live-badge ${stateClass(inspection)}`;
  elements.liveBadge.textContent = inspection === "running" ? "LIVE" : inspection === "starting" ? "STARTING" : "OFFLINE";
  elements.start.disabled = running || data.model_status !== "ready";
  elements.emptyStart.disabled = running || data.model_status !== "ready";
  elements.stop.disabled = !running;
  elements.reset.disabled = inspection === "starting";
  if (running) attachStream();
  else if (streamAttached) { elements.video.removeAttribute("src"); streamAttached = false; streamReady = false; }
  elements.placeholder.classList.toggle("hidden", inspection === "running" && streamReady);
  if (inspection === "starting") {
    elements.placeholderTitle.textContent = "Starting inspection";
    elements.placeholderCopy.textContent = "Opening the local camera and preparing inference.";
  } else if (inspection === "errored") {
    elements.placeholderTitle.textContent = "Camera unavailable";
    elements.placeholderCopy.textContent = "Check the connected camera, then start inspection again.";
  } else {
    elements.placeholderTitle.textContent = "Camera disconnected";
    elements.placeholderCopy.textContent = "Connect or start the inspection camera to begin real-time analysis.";
  }
  const fps = Number(data.fps || 0);
  elements.feedFps.textContent = `${fps.toFixed(1)} FPS`;
  const frameSize = data.frame_size;
  elements.feedResolution.textContent = Array.isArray(frameSize) ? `${frameSize[0]} × ${frameSize[1]}` : "—";
  elements.settingsResolution.textContent = elements.feedResolution.textContent;
  elements.settingsFps.textContent = `${fps.toFixed(1)} FPS`;
}

function analysisFor(event) { return event?.analysis && typeof event.analysis === "object" ? event.analysis : {}; }
function partLabel(grade) { return String(grade || "Unknown").replace("Class ", ""); }
function latestFeatureRows(event) {
  const parts = analysisFor(event).part_results || {};
  return ["Body", "Head", "Tail"].map((region) => {
    const part = parts[region] || {}; const present = part.present === true;
    const confidence = present ? formatPercent(Number(part.grade_confidence) * 100) : "Unknown";
    const weight = part.normalized_weight == null ? "" : ` · w ${formatPercent(Number(part.normalized_weight) * 100)}`;
    return [region, present ? `${partLabel(part.grade)} ${confidence}${weight}` : "Not observed (not marked missing)", present ? "detected" : "unmeasured"];
  });
}
function appendFullAnalysis(container, event) {
  const analysis = analysisFor(event); if (!Object.keys(analysis).length) return;
  const details = document.createElement("details"); details.className = "full-analysis";
  const summary = document.createElement("summary"); summary.textContent = "View full analysis"; details.append(summary);
  const body = document.createElement("div"); body.className = "full-analysis-body";
  const title = document.createElement("strong"); const finalScore = analysis.final_score == null ? "Ungraded" : formatPercent(Number(analysis.final_score) * 100);
  title.textContent = `Final verdict: ${analysis.final_grade || "Ungraded"} · ${finalScore}`; body.append(title);
  const selected = analysis.final_grade || analysis.provisional_grade;
  const results = analysis.part_results || {};
  ["Body", "Head", "Tail"].forEach((region) => {
    const part = results[region] || {}; const contribution = part.contributions?.[selected];
    const row = document.createElement("p");
    row.textContent = part.present ? `${region}: ${partLabel(part.grade)} ${formatPercent(Number(part.grade_confidence) * 100)} × ${formatPercent(Number(part.normalized_weight) * 100)} = ${formatPercent(Number(contribution) * 100)}` : `${region}: unknown—not observed, not treated as physically missing.`;
    body.append(row);
  });
  const scores = analysis.weighted_scores || {}; const scoreLine = document.createElement("p");
  scoreLine.textContent = `Weighted scores — A ${formatPercent(Number(scores["Class A"]) * 100)}, B ${formatPercent(Number(scores["Class B"]) * 100)}, C ${formatPercent(Number(scores["Class C"]) * 100)}, Rejected ${formatPercent(Number(scores.Rejected) * 100)}.`; body.append(scoreLine);
  const whole = analysis.color?.["Whole Fish"] || {}; const color = document.createElement("p");
  color.textContent = whole.status === "measured" ? `HSV measurement (Model 1 crop ROI): hue ${Number(whole.mean_hue_deg).toFixed(1)}°, saturation ${formatPercent(Number(whole.mean_saturation) * 100)}, yellow proxy ${formatPercent(Number(whole.yellow_ratio_proxy) * 100)}. It does not adjust this grade.` : "HSV measurement unavailable for this fish."; body.append(color);
  const source = document.createElement("p"); source.textContent = "Defects: none reported—the supplied checkpoints have no trained crack/yellowing/deformation labels. Missing parts: only Unknown/Not observed unless real structural evidence is supplied."; body.append(source);
  (analysis.explanation || []).forEach((text) => { const line = document.createElement("p"); line.textContent = String(text); body.append(line); });
  details.append(body); container.append(details);
}
function renderLatest(data) {
  const event = data.latest_event;
  elements.resultEmpty.classList.toggle("hidden", Boolean(event));
  elements.resultActive.classList.toggle("hidden", !event);
  elements.decision.className = `decision ${event ? "healthy" : "neutral"}`;
  elements.decision.textContent = event ? event.decision : "WAITING";
  elements.latestFeatures.replaceChildren();
  latestFeatureRows(event).forEach(([label, value, className]) => {
    const row = document.createElement("div"); row.className = "feature-row";
    const name = document.createElement("span"); name.textContent = label;
    const status = document.createElement("b"); status.textContent = value; if (className) status.classList.add(className);
    row.append(name, status); elements.latestFeatures.append(row);
  });
  appendFullAnalysis(elements.latestFeatures, event);
  if (!event) return;
  elements.resultFish.textContent = event.fish_label || `Fish #${event.track_id}`;
  elements.resultLabel.textContent = event.quality || "Ungraded";
  const analysis = analysisFor(event); const confidence = analysis.final_score == null ? (event.quality_confidence == null ? Number(event.confidence_percent) : Number(event.quality_confidence) * 100) : Number(analysis.final_score) * 100;
  elements.confidence.textContent = formatPercent(confidence);
  elements.confidenceBar.style.width = `${Math.max(0, Math.min(100, confidence || 0))}%`;
  elements.resultProcessing.textContent = formatMs(event.processing_time_ms);
  elements.resultTime.textContent = formatTime(event.timestamp);
  elements.classificationPolicy.textContent = event.quality ? `Grade is weighted from Model 2 Head/Body/Tail evidence (${analysis.evidence_completeness || "unknown"} evidence); Model 1 confidence remains traceability only.` : "No final grade yet: retained Model 2 evidence is incomplete or awaiting temporal confirmation.";
}

function kpiCard(label, value, note, emphasis = false) {
  const card = document.createElement("article"); card.className = `kpi-card${emphasis ? " emphasis" : ""}`;
  const labelEl = document.createElement("span"); labelEl.className = "kpi-label"; labelEl.append(document.createElement("i"), document.createTextNode(label));
  const valueEl = document.createElement("strong"); valueEl.textContent = String(value);
  const noteEl = document.createElement("small"); noteEl.textContent = note;
  card.append(labelEl, valueEl, noteEl); return card;
}
function renderKpis(data) {
  const counters = data.counters || {}; const grades = data.quality_counters || {}; const total = Number(counters.total || 0);
  const percentage = (name) => total ? `${((Number(grades[name] || 0) / total) * 100).toFixed(1)}% of inspected fish` : "No fish this session";
  const cards = [["Total fish", total, "Current session", true]];
  Object.entries(grades).forEach(([name, count]) => cards.push([name, count || 0, percentage(name)]));
  cards.push(["Active fish", data.active_fish_count || 0, "Live tracks"], ["Inspection rate", `${Number(data.fps || 0).toFixed(1)}`, "Frames per second"]);
  elements.kpiGrid.replaceChildren(...cards.map(([label, value, note, emphasis]) => kpiCard(label, value, note, Boolean(emphasis))));
  elements.threshold.textContent = formatPercent(Number(data.detection_confidence_threshold) * 100);
}

function renderFeatureCounters(data) {
  elements.featureCounterGrid.replaceChildren();
  Object.entries(data.feature_counters || {}).forEach(([name, detail]) => {
    const card = document.createElement("article"); card.className = `feature-counter${detail.available ? "" : " unavailable"}`;
    const title = document.createElement("span"); title.textContent = name;
    const value = document.createElement("strong"); value.textContent = detail.available ? String(detail.count || 0) : "Not measured";
    const note = document.createElement("small"); note.textContent = detail.source;
    card.append(title, value, note); elements.featureCounterGrid.append(card);
  });
}

function ensureAnalysisCharts() {
  if (partQualityChart && colorTrendChart) return;
  const grid = document.querySelector("#analytics-section .analytics-grid"); if (!grid) return;
  const makeCard = (title) => { const card = document.createElement("article"); card.className = "chart-card glass"; const heading = document.createElement("h3"); heading.textContent = title; const chart = document.createElement("div"); chart.className = "bar-chart"; card.append(heading, chart); grid.append(card); return chart; };
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
  elements.confidenceChartEmpty.classList.toggle("hidden", !hasData || confidenceAvailable);
  if (!hasData) { if (partQualityChart) partQualityChart.replaceChildren(); if (colorTrendChart) colorTrendChart.replaceChildren(); return; }
  const distribution = analytics.grade_distribution || {}; const total = Object.values(distribution).reduce((sum, value) => sum + Number(value || 0), 0);
  let offset = 0; const segments = []; const colors = { ...gradeColors };
  Object.keys(distribution).forEach((name, index) => { if (!colors[name]) colors[name] = ["#3587a4", "#be7d3f", "#7875c2"][index % 3]; });
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
  elements.throughputCaption.textContent = `${throughput.length} processed`;
  if (confidenceAvailable) elements.averageGradeConfidence.textContent = `M1 ${formatPercent(analytics.average_detection_confidence)} · M2 ${formatPercent(analytics.average_grade_confidence)}`;
}

function detectedFeatures(event) {
  const regions = [...new Set((event?.parts || []).map((part) => part.region).filter(Boolean))];
  return regions.length ? regions.join(", ") : "No Model 2 regions";
}
function openHistoryAnalysis(event) {
  let dialog = $("history-analysis-dialog");
  if (!dialog) { dialog = document.createElement("dialog"); dialog.id = "history-analysis-dialog"; dialog.className = "history-analysis-dialog"; document.body.append(dialog); }
  dialog.replaceChildren(); const close = document.createElement("button"); close.className = "icon-button"; close.type = "button"; close.textContent = "×"; close.addEventListener("click", () => dialog.close());
  const title = document.createElement("h2"); title.textContent = `${event.fish_label || "Fish"} full analysis`;
  const content = document.createElement("div"); appendFullAnalysis(content, event); dialog.append(close, title, content);
  if (typeof dialog.showModal === "function") dialog.showModal(); else dialog.setAttribute("open", "");
}
function historyCell(value, className = "") { const cell = document.createElement("td"); cell.textContent = value; if (className) cell.className = className; return cell; }
function renderRecent(history) {
  const rows = (history || []).slice(0, 10);
  elements.recentEmpty.classList.toggle("hidden", rows.length > 0); elements.recentWrap.classList.toggle("hidden", rows.length === 0); elements.recentHistoryBody.replaceChildren();
  rows.forEach((event) => {
    const confidence = event.quality_confidence == null ? event.confidence_percent : Number(event.quality_confidence) * 100;
    const row = document.createElement("tr"); row.append(historyCell(event.fish_label), historyCell(formatTime(event.timestamp)), historyCell(event.quality || "Ungraded", "grade-cell"), historyCell(formatPercent(confidence)), historyCell(detectedFeatures(event), "status-note"), historyCell(formatMs(event.processing_time_ms))); elements.recentHistoryBody.append(row);
  });
}

function renderModel(data) {
  const detector = data.model_info?.detector || {}; const segmenter = data.model_info?.quality || {}; const diagnostic = segmenter.diagnostics || {};
  elements.detectorModelName.textContent = detector.name || "Fish detector"; elements.detectorModelStatus.textContent = titleCase(detector.status); elements.detectorCheckpoint.textContent = detector.checkpoint_resolved ? "Resolved" : "Unresolved"; elements.detectorDevice.textContent = detector.device || "—"; elements.detectorThreshold.textContent = formatPercent(Number(detector.confidence_threshold) * 100);
  elements.segmenterModelName.textContent = segmenter.name || "Quality Model"; elements.segmenterModelStatus.textContent = titleCase(segmenter.status); elements.segmenterCheckpoint.textContent = segmenter.checkpoint_resolved ? "Resolved" : "Unresolved"; elements.segmenterDevice.textContent = segmenter.device || "—"; elements.segmenterInference.textContent = diagnostic.last_inference_ms == null ? "No measurement" : `${diagnostic.last_inference_ms} ms`;
}

function syncSettings(data) {
  const threshold = Math.round(Number(data.detection_confidence_threshold || 0) * 100);
  elements.slider.value = String(threshold); elements.settingsThreshold.textContent = `${threshold}%`;
  const qualityThreshold = Math.round(Number(data.quality_confidence_threshold || 0) * 100);
  elements.qualitySlider.value = String(qualityThreshold); elements.settingsQualityThreshold.textContent = `${qualityThreshold}%`;
  const display = data.display_settings || {};
  Object.entries(overlayControls).forEach(([name, controls]) => controls.forEach((control) => { if (control) control.checked = Boolean(display[name]); }));
}

function render(data) {
  latestStatus = data; defaultConfidence = Number(data.default_confidence_threshold ?? defaultConfidence); defaultQualityConfidence = Number(data.default_quality_confidence_threshold ?? defaultQualityConfidence); renderAlert(data); renderCamera(data); renderLatest(data); renderKpis(data); renderFeatureCounters(data); renderAnalytics(data.analytics); renderRecent(data.recent_history); renderModel(data); syncSettings(data);
  if (selectedView === "history") loadHistory();
}

function renderBackendUnavailable() {
  elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Backend unavailable"; elements.alertCopy.textContent = "Cannot reach the local Python backend."; elements.alertDetails.textContent = "Start the local server, then reload this page.";
  [elements.start, elements.emptyStart, elements.stop, elements.reset, elements.analyze].forEach((button) => { button.disabled = true; });
}
async function getStatus() {
  try { const response = await fetch("/api/status", { cache: "no-store" }); if (!response.ok) throw new Error(`HTTP ${response.status}`); render(await response.json()); }
  catch (_) { renderBackendUnavailable(); }
}
async function postControl(path) {
  [elements.start, elements.emptyStart, elements.stop, elements.reset].forEach((button) => { button.disabled = true; });
  try { const response = await fetch(path, { method: "POST" }); if (!response.ok) throw new Error(`HTTP ${response.status}`); render(await response.json()); }
  catch (_) { renderBackendUnavailable(); }
}
async function postSettings(payload) {
  try { const response = await fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`); render(data); }
  catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Settings not applied"; elements.alertCopy.textContent = "The active configuration could not be updated."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown settings error."; }
}

function showView(view) {
  selectedView = view;
  Object.keys(viewMeta).forEach((name) => $("view-" + name).classList.toggle("hidden", name !== view));
  const [title, subtitle] = viewMeta[view]; elements.pageTitle.textContent = title; elements.pageSubtitle.textContent = subtitle;
  document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  if (view === "history") loadHistory(true);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function openSettings() { elements.settingsDrawer.classList.add("open"); elements.settingsDrawer.setAttribute("aria-hidden", "false"); elements.scrim.classList.remove("hidden"); }
function closeSettings() { elements.settingsDrawer.classList.remove("open"); elements.settingsDrawer.setAttribute("aria-hidden", "true"); if (elements.exportModal.classList.contains("hidden")) elements.scrim.classList.add("hidden"); }
function openExport() { elements.exportModal.classList.remove("hidden"); elements.scrim.classList.remove("hidden"); }
function closeExport() { elements.exportModal.classList.add("hidden"); if (!elements.settingsDrawer.classList.contains("open")) elements.scrim.classList.add("hidden"); }

async function loadHistory(resetPage = false) {
  if (resetPage) historyState.page = 1;
  const query = new URLSearchParams({ page: String(historyState.page), page_size: "25" });
  if (elements.historySearch.value.trim()) query.set("search", elements.historySearch.value.trim());
  if (elements.historyGrade.value) query.set("grade", elements.historyGrade.value);
  if (elements.historyFeature.value) query.set("feature", elements.historyFeature.value);
  if (elements.historyConfidence.value) query.set("min_confidence", elements.historyConfidence.value);
  try {
    const response = await fetch(`/api/history?${query}`); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "History unavailable");
    historyState.totalPages = data.total_pages; historyState.page = data.page; elements.historyPage.textContent = `Page ${data.page} of ${data.total_pages}`; elements.historyPrev.disabled = data.page <= 1; elements.historyNext.disabled = data.page >= data.total_pages;
    const rows = data.items || []; elements.historyEmpty.classList.toggle("hidden", rows.length > 0); elements.historyWrap.classList.toggle("hidden", rows.length === 0); elements.historyBody.replaceChildren();
    rows.forEach((event) => { const confidence = event.quality_confidence == null ? event.confidence_percent : Number(event.quality_confidence) * 100; const parts = event.parts || []; const regions = [...new Set(parts.map((part) => part.region).filter(Boolean))].join(", ") || "—"; const classes = [...new Set(parts.map((part) => part.source_class_name).filter(Boolean))].join(", ") || "—"; const row = document.createElement("tr"); const fish = historyCell(event.fish_label); fish.tabIndex = 0; fish.title = "Open full analysis"; fish.classList.add("history-open"); fish.addEventListener("click", () => openHistoryAnalysis(event)); fish.addEventListener("keydown", (key) => { if (key.key === "Enter") openHistoryAnalysis(event); }); row.append(fish, historyCell(formatTime(event.timestamp)), historyCell(event.quality || "Ungraded", "grade-cell"), historyCell(formatPercent(confidence)), historyCell(formatPercent(event.confidence_percent)), historyCell(regions, "status-note"), historyCell(classes, "status-note"), historyCell(formatMs(event.processing_time_ms))); elements.historyBody.append(row); });
  } catch (error) { elements.historyEmpty.textContent = error instanceof Error ? error.message : "History unavailable."; elements.historyEmpty.classList.remove("hidden"); elements.historyWrap.classList.add("hidden"); }
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
function toggleUpload(show) { elements.uploadWorkspace.classList.toggle("hidden", !show); if (show) elements.uploadWorkspace.scrollIntoView({ behavior: "smooth", block: "start" }); }

document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll(".nav-item[data-anchor]").forEach((button) => button.addEventListener("click", () => { showView("live"); setTimeout(() => { $(button.dataset.anchor).scrollIntoView({ behavior: "smooth", block: "start" }); document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === button)); }, 30); }));
elements.sidebarToggle.addEventListener("click", () => elements.shell.classList.toggle("sidebar-collapsed"));
elements.start.addEventListener("click", () => postControl("/api/inspection/start")); elements.emptyStart.addEventListener("click", () => postControl("/api/inspection/start")); elements.stop.addEventListener("click", () => postControl("/api/inspection/stop"));
elements.reset.addEventListener("click", () => { if (!hasEvents(latestStatus) || window.confirm("Reset this session? Current counters and inspection history will be cleared.")) postControl("/api/session/reset"); });
elements.uploadMode.addEventListener("click", () => toggleUpload(true)); elements.closeUpload.addEventListener("click", () => toggleUpload(false));
elements.uploadInput.addEventListener("change", () => { selectedUpload = elements.uploadInput.files?.[0] || null; elements.uploadResults.classList.add("hidden"); if (!selectedUpload) { elements.uploadStatus.textContent = "No image selected"; elements.uploadImage.classList.add("hidden"); elements.analyze.disabled = true; return; } elements.uploadStatus.textContent = "Ready"; elements.uploadNote.textContent = `${selectedUpload.name} is ready for separate local analysis.`; const reader = new FileReader(); reader.onload = () => { elements.uploadImage.src = reader.result; elements.uploadImage.classList.remove("hidden"); }; reader.readAsDataURL(selectedUpload); elements.analyze.disabled = !latestStatus?.upload_analysis_available; });
elements.analyze.addEventListener("click", async () => { if (!selectedUpload || uploadInFlight) return; uploadInFlight = true; elements.analyze.disabled = true; elements.uploadStatus.textContent = "Analyzing"; elements.uploadNote.textContent = "Running Model 1, then Model 2 on each detected fish crop."; try { const response = await fetch("/api/analyze-image", { method: "POST", headers: { "Content-Type": selectedUpload.type || "application/octet-stream", "X-Filename": selectedUpload.name }, body: selectedUpload }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`); const isPartPreview = payload.analysis_mode === "part_preview"; elements.uploadImage.src = payload.annotated_image; elements.uploadFishCount.textContent = String(isPartPreview ? payload.parts_detected : payload.fish_detected); elements.uploadCountLabel.textContent = isPartPreview ? "Parts detected" : "Fish found"; renderUploadQuality(payload.quality_counters); elements.uploadResults.classList.remove("hidden"); elements.uploadStatus.textContent = "Complete"; elements.uploadNote.textContent = isPartPreview ? "Raw part-preview test complete. It does not create fish counts or live inspection records." : payload.model2_available ? "Model 1 fish detections and Model 2 part/grade evidence are shown." : "Detection complete; the Quality Model is unavailable."; } catch (error) { elements.uploadStatus.textContent = "Error"; elements.uploadNote.textContent = error instanceof Error ? error.message : "Image analysis failed."; } finally { uploadInFlight = false; elements.analyze.disabled = !selectedUpload || !latestStatus?.upload_analysis_available; } });
function renderUploadQuality(counters) { const grid = $("upload-quality-counter-grid"); grid.replaceChildren(); Object.entries(counters || {}).forEach(([name, value]) => { const card = document.createElement("article"); card.className = "feature-counter"; card.append(Object.assign(document.createElement("span"), { textContent: name }), Object.assign(document.createElement("strong"), { textContent: String(value) }), Object.assign(document.createElement("small"), { textContent: "Image analysis only" })); grid.append(card); }); }

elements.viewAll.addEventListener("click", () => showView("history")); elements.historyFilter.addEventListener("click", () => loadHistory(true)); elements.historyPrev.addEventListener("click", () => { if (historyState.page > 1) { historyState.page -= 1; loadHistory(); } }); elements.historyNext.addEventListener("click", () => { if (historyState.page < historyState.totalPages) { historyState.page += 1; loadHistory(); } });
elements.settingsButton.addEventListener("click", openSettings); elements.sidebarSettings.addEventListener("click", openSettings); elements.closeSettings.addEventListener("click", closeSettings); elements.openConfidence.addEventListener("click", openSettings); elements.scrim.addEventListener("click", () => { closeSettings(); closeExport(); });
elements.exportButton.addEventListener("click", openExport); elements.sidebarExport.addEventListener("click", openExport); elements.closeExport.addEventListener("click", closeExport); elements.cancelExport.addEventListener("click", closeExport);
elements.slider.addEventListener("input", () => setConfidence(elements.slider.value)); elements.slider.addEventListener("change", () => setConfidence(elements.slider.value, true)); elements.confidenceMinus.addEventListener("click", () => setConfidence(Number(elements.slider.value) - 1, true)); elements.confidencePlus.addEventListener("click", () => setConfidence(Number(elements.slider.value) + 1, true)); elements.resetConfidence.addEventListener("click", () => setConfidence(defaultConfidence * 100, true));
elements.qualitySlider.addEventListener("input", () => setQualityConfidence(elements.qualitySlider.value)); elements.qualitySlider.addEventListener("change", () => setQualityConfidence(elements.qualitySlider.value, true)); elements.qualityConfidenceMinus.addEventListener("click", () => setQualityConfidence(Number(elements.qualitySlider.value) - 1, true)); elements.qualityConfidencePlus.addEventListener("click", () => setQualityConfidence(Number(elements.qualitySlider.value) + 1, true)); elements.resetQualityConfidence.addEventListener("click", () => setQualityConfidence(defaultQualityConfidence * 100, true));
Object.entries(overlayControls).forEach(([name, controls]) => controls.forEach((control) => control.addEventListener("change", () => setOverlay(name, control.checked))));
document.querySelectorAll("input[name=export-range]").forEach((input) => input.addEventListener("change", () => { elements.dateRange.classList.toggle("hidden", document.querySelector("input[name=export-range]:checked").value !== "custom"); }));
elements.exportForm.addEventListener("submit", async (event) => { event.preventDefault(); const button = $("confirm-export-button"); const fields = [...document.querySelectorAll(".export-fields input:checked")].map((input) => input.value); if (!fields.length) return; const payload = { format: document.querySelector("input[name=export-format]:checked").value, range: document.querySelector("input[name=export-range]:checked").value, fields, complete_analysis: true, start_date: elements.exportStartDate.value || null, end_date: elements.exportEndDate.value || null }; button.disabled = true; try { const response = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); if (!response.ok) { const error = await response.json(); throw new Error(error.detail || "Export failed."); } const blob = await response.blob(); const anchor = document.createElement("a"); const disposition = response.headers.get("content-disposition") || ""; const name = disposition.match(/filename="?([^";]+)"?/)?.[1] || `sardinella_inspection.${payload.format}`; anchor.href = URL.createObjectURL(blob); anchor.download = name; document.body.append(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(anchor.href); closeExport(); } catch (error) { elements.alert.classList.remove("hidden"); elements.alertTitle.textContent = "Export unavailable"; elements.alertCopy.textContent = "Inspection data could not be exported."; elements.alertDetails.textContent = error instanceof Error ? error.message : "Unknown export error."; } finally { button.disabled = false; } });

getStatus(); setInterval(getStatus, 900);
