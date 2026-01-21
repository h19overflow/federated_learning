"""Results table section builder for batch reports."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from federated_pneumonia_detection.src.control.report_generation.internals.constants import (
    BRAND_COLOR,
    DANGER_COLOR,
    LIGHT_BG,
    SUCCESS_COLOR,
    WARNING_COLOR,
)


def create_results_table(story: list, styles, results: list) -> None:
    """Create detailed results table.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        results: List of prediction result dictionaries
    """
    story.append(Paragraph("Detailed Results", styles["SectionHeader"]))
    story.append(Spacer(1, 8))

    results_header = [
        Paragraph(
            "<b>#</b>",
            ParagraphStyle(
                "TH",
                fontSize=8,
                textColor=colors.white,
                alignment=TA_CENTER,
            ),
        ),
        Paragraph(
            "<b>Filename</b>",
            ParagraphStyle("TH", fontSize=8, textColor=colors.white),
        ),
        Paragraph(
            "<b>Classification</b>",
            ParagraphStyle(
                "TH",
                fontSize=8,
                textColor=colors.white,
                alignment=TA_CENTER,
            ),
        ),
        Paragraph(
            "<b>Confidence</b>",
            ParagraphStyle(
                "TH",
                fontSize=8,
                textColor=colors.white,
                alignment=TA_CENTER,
            ),
        ),
        Paragraph(
            "<b>Pneumonia %</b>",
            ParagraphStyle(
                "TH",
                fontSize=8,
                textColor=colors.white,
                alignment=TA_CENTER,
            ),
        ),
        Paragraph(
            "<b>Normal %</b>",
            ParagraphStyle(
                "TH",
                fontSize=8,
                textColor=colors.white,
                alignment=TA_CENTER,
            ),
        ),
    ]
    results_data = [results_header]

    for i, result in enumerate(results[:100], 1):
        row = _build_result_row(i, result)
        results_data.append(row)

    results_table = Table(
        results_data,
        colWidths=[
            0.35 * inch,
            2.1 * inch,
            1.1 * inch,
            0.85 * inch,
            0.9 * inch,
            0.85 * inch,
        ],
    )
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ],
        ),
    )
    story.append(results_table)

    if len(results) > 100:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"<i>Note: Showing first 100 of {len(results)} results. "
                "Export to CSV for complete data.</i>",
                styles["Disclaimer"],
            ),
        )

    story.append(Spacer(1, 16))


def _build_result_row(index: int, result: dict) -> list:
    """Build a single row for the results table."""
    if result.get("success"):
        pred = result.get("prediction", {})
        pred_class = pred.get("predicted_class", "N/A")
        conf = pred.get("confidence", 0) * 100
        pneumonia_prob = pred.get("pneumonia_probability", 0) * 100
        normal_prob = pred.get("normal_probability", 0) * 100

        class_color = WARNING_COLOR if pred_class == "PNEUMONIA" else SUCCESS_COLOR
        class_para = Paragraph(
            f"<b>{pred_class}</b>",
            ParagraphStyle(
                "Class",
                fontSize=8,
                textColor=class_color,
                alignment=TA_CENTER,
            ),
        )
    else:
        conf = 0
        pneumonia_prob = 0
        normal_prob = 0
        class_para = Paragraph(
            "<b>ERROR</b>",
            ParagraphStyle(
                "Class",
                fontSize=8,
                textColor=DANGER_COLOR,
                alignment=TA_CENTER,
            ),
        )

    filename = result.get("filename", "Unknown")
    if len(filename) > 28:
        filename = filename[:25] + "..."

    return [
        Paragraph(str(index), ParagraphStyle("TD", fontSize=8, alignment=TA_CENTER)),
        Paragraph(filename, ParagraphStyle("TD", fontSize=8)),
        class_para,
        Paragraph(
            f"{conf:.1f}%",
            ParagraphStyle("TD", fontSize=8, alignment=TA_CENTER),
        ),
        Paragraph(
            f"{pneumonia_prob:.1f}%",
            ParagraphStyle("TD", fontSize=8, alignment=TA_CENTER),
        ),
        Paragraph(
            f"{normal_prob:.1f}%",
            ParagraphStyle("TD", fontSize=8, alignment=TA_CENTER),
        ),
    ]
