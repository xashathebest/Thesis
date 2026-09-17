"""CSV/XLSX exports for traceable batch hand-off."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..database.db import InspectionDatabase


def export_batch_excel(database: InspectionDatabase, batch_id: str, output_dir: Path) -> tuple[Path, Path]:
    """Write one-row-per-fish Excel/CSV plus a Batch Summary worksheet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = database.records_for_batch(batch_id)
    individual = pd.DataFrame(records)
    summary = pd.DataFrame([database.summarize_rows(records)])
    safe_batch_id = batch_id.replace("/", "-").replace("\\", "-")
    workbook_path = output_dir / f"Fish_Inspection_{safe_batch_id}.xlsx"
    csv_path = output_dir / f"Fish_Inspection_{safe_batch_id}.csv"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        individual.to_excel(writer, sheet_name="Individual Fish", index=False)
        summary.to_excel(writer, sheet_name="Batch Summary", index=False)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            for column in worksheet.columns:
                letter = column[0].column_letter
                worksheet.column_dimensions[letter].width = min(42, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
    individual.to_csv(csv_path, index=False)
    return workbook_path, csv_path
