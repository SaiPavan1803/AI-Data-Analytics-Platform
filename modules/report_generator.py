"""
report_generator.py
-------------------
Builds a multi-section PDF report using ReportLab. Plotly figures are
rendered to PNG via kaleido and embedded in the document.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _fig_to_png_bytes(fig) -> Optional[bytes]:
    """Render a Plotly figure to PNG bytes. Returns None on failure."""
    try:
        return fig.to_image(format="png", width=900, height=500, scale=2)
    except Exception:
        return None


def _df_to_table(df: pd.DataFrame, max_rows: int = 15) -> Table:
    df = df.head(max_rows).copy()
    df = df.astype(str)
    data = [list(df.columns)] + df.values.tolist()
    tbl = Table(data, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a93")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#eef2f7")]),
    ]))
    return tbl


def build_report(
    output_path: str,
    *,
    dataset_name: str,
    overview: dict,
    explanation: str,
    insights: List[str],
    figures: Iterable[Tuple[str, object]],
    anomaly_table: Optional[pd.DataFrame] = None,
    anomaly_notes: Optional[List[str]] = None,
    forecast_table: Optional[pd.DataFrame] = None,
    forecast_figure=None,
    recommendations: Optional[List[str]] = None,
) -> str:
    """Build a PDF report and return the output_path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"AI Data Analyst Report — {dataset_name}",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"],
                        textColor=colors.HexColor("#1f3a93"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"],
                        textColor=colors.HexColor("#1f3a93"))
    body = styles["BodyText"]

    story = []

    # Title
    story.append(Paragraph("AI Data Analyst Report", h1))
    story.append(Paragraph(
        f"Dataset: <b>{dataset_name}</b><br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body))
    story.append(Spacer(1, 0.5 * cm))

    # 1. Overview
    story.append(Paragraph("1. Dataset Overview", h2))
    ov = (f"Rows: {overview.get('rows')}  |  "
          f"Columns: {overview.get('cols')}  |  "
          f"Duplicates: {overview.get('duplicate_rows', 0)}  |  "
          f"Memory: {overview.get('memory_kb', 0)} KB")
    story.append(Paragraph(ov, body))
    if overview.get("missing_values"):
        miss = ", ".join(f"{k} ({v})" for k, v in overview["missing_values"].items())
        story.append(Paragraph(f"<b>Missing values:</b> {miss}", body))
    story.append(Spacer(1, 0.3 * cm))

    # 2. Summary / explanation
    story.append(Paragraph("2. Data Summary", h2))
    story.append(Paragraph(explanation.replace("\n", "<br/>"), body))
    story.append(Spacer(1, 0.3 * cm))

    # 3. Key insights
    story.append(Paragraph("3. Key Insights", h2))
    for ins in insights:
        story.append(Paragraph("• " + ins.replace("**", ""), body))
    story.append(Spacer(1, 0.3 * cm))

    # 4. Charts
    story.append(PageBreak())
    story.append(Paragraph("4. Charts", h2))
    for title, fig in figures:
        png = _fig_to_png_bytes(fig)
        story.append(Paragraph(f"<b>{title}</b>", body))
        if png:
            story.append(Image(io.BytesIO(png), width=16 * cm, height=9 * cm))
        else:
            story.append(Paragraph("(Chart rendering unavailable — install kaleido.)", body))
        story.append(Spacer(1, 0.3 * cm))

    # 5. Anomalies
    if anomaly_table is not None and not anomaly_table.empty:
        story.append(PageBreak())
        story.append(Paragraph("5. Anomalies", h2))
        story.append(_df_to_table(anomaly_table.reset_index().head(10)))
        story.append(Spacer(1, 0.2 * cm))
        for note in (anomaly_notes or []):
            story.append(Paragraph("• " + note, body))

    # 6. Forecast
    if forecast_table is not None and not forecast_table.empty:
        story.append(PageBreak())
        story.append(Paragraph("6. Forecast Results", h2))
        if forecast_figure is not None:
            png = _fig_to_png_bytes(forecast_figure)
            if png:
                story.append(Image(io.BytesIO(png), width=16 * cm, height=9 * cm))
                story.append(Spacer(1, 0.2 * cm))
        story.append(_df_to_table(forecast_table))

    # 7. Recommendations
    if recommendations:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("7. Recommendations", h2))
        for rec in recommendations:
            story.append(Paragraph("• " + rec, body))

    doc.build(story)
    return output_path
