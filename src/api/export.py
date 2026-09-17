"""Session-history filtering and explainable CSV/XLSX export helpers."""

from __future__ import annotations

import csv
from datetime import date, datetime
from io import BytesIO, StringIO
from typing import Iterable, Mapping


EXPORT_FIELDS = {
    "fish_id": "Fish ID", "timestamp": "Timestamp", "grade": "Final Grade",
    "final_weighted_score": "Final Weighted Score (%)", "detection_confidence": "Model 1 Detection Confidence (%)",
    "grade_confidence": "Final Grade Confidence (%)", "body_grade": "Body Grade",
    "body_grade_confidence": "Body Grade Confidence (%)", "body_weight": "Body Weight",
    "body_weighted_contribution": "Body Winning Contribution (%)", "head_grade": "Head Grade",
    "head_grade_confidence": "Head Grade Confidence (%)", "head_weight": "Head Weight",
    "head_weighted_contribution": "Head Winning Contribution (%)", "tail_grade": "Tail Grade",
    "tail_grade_confidence": "Tail Grade Confidence (%)", "tail_weight": "Tail Weight",
    "tail_weighted_contribution": "Tail Winning Contribution (%)", "class_a_weighted_score": "Class A Weighted Score (%)",
    "class_b_weighted_score": "Class B Weighted Score (%)", "class_c_weighted_score": "Class C Weighted Score (%)",
    "rejected_weighted_score": "Rejected Weighted Score (%)", "evidence_completeness": "Evidence Completeness",
    "head_presence": "Head Presence Status", "body_presence": "Body Presence Status", "tail_presence": "Tail Presence Status",
    "body_mean_hue": "Body Mean Hue (deg)", "body_median_hue": "Body Median Hue (deg)", "body_mean_saturation": "Body Mean Saturation (%)", "body_mean_value": "Body Mean Value (%)", "body_yellow_ratio": "Body Yellow Ratio Proxy (%)",
    "head_mean_hue": "Head Mean Hue (deg)", "head_median_hue": "Head Median Hue (deg)", "head_mean_saturation": "Head Mean Saturation (%)", "head_mean_value": "Head Mean Value (%)", "head_yellow_ratio": "Head Yellow Ratio Proxy (%)",
    "tail_mean_hue": "Tail Mean Hue (deg)", "tail_median_hue": "Tail Median Hue (deg)", "tail_mean_saturation": "Tail Mean Saturation (%)", "tail_mean_value": "Tail Mean Value (%)", "tail_yellow_ratio": "Tail Yellow Ratio Proxy (%)",
    "whole_fish_mean_hue": "Whole Fish Mean Hue (deg)", "whole_fish_median_hue": "Whole Fish Median Hue (deg)", "whole_fish_mean_saturation": "Whole Fish Mean Saturation (%)", "whole_fish_mean_value": "Whole Fish Mean Value (%)", "whole_fish_yellow_ratio": "Whole Fish Yellow Ratio Proxy (%)",
    "detected_regions": "Detected Regions", "model2_classes": "Model 2 Classes",
    "adjustment_applied": "Adjustment Applied", "adjustment_reason": "Adjustment Reason",
    "override_applied": "Override Applied", "override_reason": "Override Reason",
    "final_explanation": "Final Explanation", "processing_time": "Processing Time (ms)",
}
DEFAULT_EXPORT_FIELDS = tuple(EXPORT_FIELDS)


def parse_timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _number(value: object, *, scale: float = 1.0) -> float | str:
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return ""
    return round(number, 4) if number == number and abs(number) != float("inf") else ""


def _part_values(event: Mapping[str, object], key: str) -> str:
    parts = event.get("parts")
    if not isinstance(parts, list):
        return ""
    return ", ".join(sorted({str(part.get(key, "")) for part in parts if isinstance(part, dict) and part.get(key)}))


def _analysis(event: Mapping[str, object]) -> dict[str, object]:
    value = event.get("analysis")
    return dict(value) if isinstance(value, dict) else {}


def _part_analysis(analysis: Mapping[str, object], region: str) -> dict[str, object]:
    parts = analysis.get("part_results")
    value = parts.get(region) if isinstance(parts, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def _color_analysis(analysis: Mapping[str, object], region: str) -> dict[str, object]:
    colors = analysis.get("color")
    value = colors.get(region) if isinstance(colors, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def inspection_row(event: dict[str, object]) -> dict[str, object]:
    """Map one completed fish to numeric, reconstructable export values."""

    analysis = _analysis(event)
    weighted = analysis.get("weighted_scores") if isinstance(analysis.get("weighted_scores"), dict) else {}
    final_grade = event.get("quality") or analysis.get("final_grade") or "Ungraded"
    final_score = analysis.get("final_score") if analysis.get("final_score") is not None else event.get("quality_confidence")
    row: dict[str, object] = {
        "fish_id": event.get("fish_label", f"Fish #{event.get('track_id', '')}"),
        "timestamp": event.get("timestamp", ""), "grade": final_grade,
        "final_weighted_score": _number(final_score, scale=100),
        "detection_confidence": _number(event.get("final_confidence"), scale=100),
        "grade_confidence": _number(event.get("quality_confidence"), scale=100),
        "class_a_weighted_score": _number(weighted.get("Class A") if isinstance(weighted, dict) else None, scale=100),
        "class_b_weighted_score": _number(weighted.get("Class B") if isinstance(weighted, dict) else None, scale=100),
        "class_c_weighted_score": _number(weighted.get("Class C") if isinstance(weighted, dict) else None, scale=100),
        "rejected_weighted_score": _number(weighted.get("Rejected") if isinstance(weighted, dict) else None, scale=100),
        "evidence_completeness": analysis.get("evidence_completeness", ""),
        "detected_regions": _part_values(event, "region"), "model2_classes": _part_values(event, "source_class_name"),
        "adjustment_applied": "Yes" if analysis.get("adjustments") else "No",
        "adjustment_reason": "; ".join(str(item.get("reason", "")) for item in analysis.get("adjustments", ()) if isinstance(item, dict)),
        "override_applied": "Yes" if isinstance(analysis.get("override"), dict) and analysis["override"].get("applied") else "No",
        "override_reason": analysis.get("override", {}).get("reason", "") if isinstance(analysis.get("override"), dict) else "",
        "final_explanation": " ".join(str(item) for item in analysis.get("explanation", ()) if item),
        "processing_time": event.get("processing_time_ms") if event.get("processing_time_ms") is not None else "",
    }
    for region, prefix in (("Body", "body"), ("Head", "head"), ("Tail", "tail")):
        part, color = _part_analysis(analysis, region), _color_analysis(analysis, region)
        contributions = part.get("contributions") if isinstance(part.get("contributions"), dict) else {}
        row[f"{prefix}_grade"] = part.get("grade", "")
        row[f"{prefix}_grade_confidence"] = _number(part.get("grade_confidence"), scale=100)
        row[f"{prefix}_weight"] = _number(part.get("weight"))
        row[f"{prefix}_weighted_contribution"] = _number(contributions.get(str(final_grade)) if isinstance(contributions, dict) else None, scale=100)
        row[f"{prefix}_presence"] = part.get("missing_status", "unknown_not_observed")
        row[f"{prefix}_mean_hue"] = _number(color.get("mean_hue_deg"))
        row[f"{prefix}_median_hue"] = _number(color.get("median_hue_deg"))
        row[f"{prefix}_mean_saturation"] = _number(color.get("mean_saturation"), scale=100)
        row[f"{prefix}_mean_value"] = _number(color.get("mean_value"), scale=100)
        row[f"{prefix}_yellow_ratio"] = _number(color.get("yellow_ratio_proxy"), scale=100)
    whole = _color_analysis(analysis, "Whole Fish")
    row["whole_fish_mean_hue"] = _number(whole.get("mean_hue_deg"))
    row["whole_fish_median_hue"] = _number(whole.get("median_hue_deg"))
    row["whole_fish_mean_saturation"] = _number(whole.get("mean_saturation"), scale=100)
    row["whole_fish_mean_value"] = _number(whole.get("mean_value"), scale=100)
    row["whole_fish_yellow_ratio"] = _number(whole.get("yellow_ratio_proxy"), scale=100)
    return row


def filter_events(events: Iterable[dict[str, object]], *, range_name: str = "current_session", start_date: str | None = None, end_date: str | None = None) -> list[dict[str, object]]:
    values = list(events)
    if range_name == "current_session":
        return values
    if range_name == "today":
        today = date.today()
        return [event for event in values if (parsed := parse_timestamp(event.get("timestamp"))) and parsed.date() == today]
    if range_name != "custom":
        raise ValueError("Unsupported export range.")
    try:
        lower = date.fromisoformat(start_date) if start_date else None
        upper = date.fromisoformat(end_date) if end_date else None
    except ValueError as exc:
        raise ValueError("Custom export dates must use YYYY-MM-DD.") from exc
    if lower and upper and lower > upper:
        raise ValueError("The start date must not be after the end date.")
    result = []
    for event in values:
        parsed = parse_timestamp(event.get("timestamp"))
        if parsed is not None and (not lower or parsed.date() >= lower) and (not upper or parsed.date() <= upper):
            result.append(event)
    return result


def make_csv(events: Iterable[dict[str, object]], fields: Iterable[str]) -> bytes:
    selected = tuple(fields)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[EXPORT_FIELDS[field] for field in selected], extrasaction="ignore")
    writer.writeheader()
    for event in events:
        row = inspection_row(event)
        writer.writerow({EXPORT_FIELDS[field]: row[field] for field in selected})
    return buffer.getvalue().encode("utf-8-sig")


def _style_header(sheet) -> None:
    from openpyxl.styles import Font, PatternFill
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2563EB")


def make_xlsx(events: Iterable[dict[str, object]], fields: Iterable[str], summary: dict[str, object], grading_rules: Mapping[str, object] | None = None) -> bytes:
    """Create Fish Inspections, Session Summary, Feature Summary, and rule sheets."""

    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Excel export requires openpyxl. Install the project requirements.") from exc
    selected, event_rows = tuple(fields), list(events)
    workbook = Workbook()
    inspections = workbook.active
    inspections.title = "Fish Inspections"
    inspections.append([EXPORT_FIELDS[field] for field in selected])
    _style_header(inspections)
    for event in event_rows:
        row = inspection_row(event)
        inspections.append([row[field] for field in selected])
    inspections.freeze_panes = "A2"
    for column in inspections.columns:
        inspections.column_dimensions[column[0].column_letter].width = min(40, max(14, max(len(str(cell.value or "")) for cell in column) + 2))

    summary_sheet = workbook.create_sheet("Session Summary")
    summary_sheet.append(["Metric", "Value"])
    _style_header(summary_sheet)
    for label, value in summary.items():
        summary_sheet.append([label, value])
    summary_sheet.column_dimensions["A"].width, summary_sheet.column_dimensions["B"].width = 38, 24

    feature_sheet = workbook.create_sheet("Feature Summary")
    feature_sheet.append(["Feature", "Count", "Percentage of fish", "Source"])
    _style_header(feature_sheet)
    total = len(event_rows)
    for region in ("Head", "Body", "Tail"):
        count = sum(int(_part_analysis(_analysis(event), region).get("present") is True) for event in event_rows)
        feature_sheet.append([f"{region} detected", count, round(count / total * 100, 2) if total else 0.0, "Model 2 part detector"])
    feature_sheet.append(["Colour measurements", total, 100.0 if total else 0.0, "HSV proxy; no active grading rule"])
    feature_sheet.append(["Trained defect labels", 0, 0.0, "None in supplied Model 1/Model 2 checkpoints"])
    for column, width in {"A": 28, "B": 14, "C": 22, "D": 52}.items(): feature_sheet.column_dimensions[column].width = width

    rules = dict(grading_rules or {})
    weights = rules.get("part_weights") if isinstance(rules.get("part_weights"), dict) else {}
    rule_sheet = workbook.create_sheet("Grade Calculation Rules")
    rule_sheet.append(["Rule", "Value", "Status"])
    _style_header(rule_sheet)
    for region in ("Body", "Head", "Tail"): rule_sheet.append([f"{region} weight", weights.get(region, ""), "Active"])
    rule_sheet.append(["Aggregation", "Highest Model 2 confidence per region/grade per frame; mean over observed frames", "Active"])
    rule_sheet.append(["Unavailable part", "Exclude and renormalize available part weights; do not mark physically missing", "Active"])
    rule_sheet.append(["Body required", rules.get("require_body", True), "Active"])
    rule_sheet.append(["Minimum final score", rules.get("minimum_final_score", ""), "Active"])
    rule_sheet.append(["Rejected override threshold", rules.get("rejected_override_threshold", "None"), "Disabled" if rules.get("rejected_override_threshold") is None else "Configured"])
    rule_sheet.append(["Color adjustments", rules.get("color_adjustments_enabled", False), "No validated rule configured"])
    rule_sheet.append(["Defect adjustments", rules.get("defect_adjustments_enabled", False), "No trained defect label or validated rule configured"])
    rule_sheet.column_dimensions["A"].width, rule_sheet.column_dimensions["B"].width, rule_sheet.column_dimensions["C"].width = 28, 92, 42
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()
