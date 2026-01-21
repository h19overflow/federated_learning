"""Executive summary section builder for batch reports."""

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


def _create_metric_box(label: str, value: str, color) -> Table:
    """Create a styled metric box for the summary."""
    return Table(
        [
            [
                Paragraph(
                    f"<b>{value}</b>",
                    ParagraphStyle(
                        "MetricValue",
                        fontSize=18,
                        textColor=color,
                        alignment=TA_CENTER,
                    ),
                ),
            ],
            [
                Paragraph(
                    label,
                    ParagraphStyle(
                        "MetricLabel",
                        fontSize=8,
                        textColor=colors.gray,
                        alignment=TA_CENTER,
                    ),
                ),
            ],
        ],
        colWidths=[1.4 * inch],
    )


def create_executive_summary(story: list, styles, summary_stats: dict) -> None:
    """Create executive summary section with key metrics.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        summary_stats: Dictionary with aggregate statistics
    """
    story.append(Paragraph("Executive Summary", styles["SectionHeader"]))
    story.append(Spacer(1, 8))

    total = summary_stats.get("total_images", 0)
    successful = summary_stats.get("successful", 0)
    pneumonia_count = summary_stats.get("pneumonia_count", 0)
    normal_count = summary_stats.get("normal_count", 0)
    avg_confidence = summary_stats.get("avg_confidence", 0) * 100
    high_risk = summary_stats.get("high_risk_count", 0)

    pneumonia_pct = (pneumonia_count / max(successful, 1)) * 100
    normal_pct = (normal_count / max(successful, 1)) * 100

    metrics_row = [
        _create_metric_box("Total Analyzed", str(total), BRAND_COLOR),
        _create_metric_box("Pneumonia", f"{pneumonia_count}", WARNING_COLOR),
        _create_metric_box("Normal", f"{normal_count}", SUCCESS_COLOR),
        _create_metric_box("Avg Confidence", f"{avg_confidence:.0f}%", BRAND_COLOR),
        _create_metric_box(
            "High Risk",
            str(high_risk),
            DANGER_COLOR if high_risk > 0 else colors.gray,
        ),
    ]

    metrics_table = Table([metrics_row], colWidths=[1.4 * inch] * 5)
    metrics_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, 0), 1, BRAND_COLOR),
                ("BOX", (1, 0), (1, 0), 1, WARNING_COLOR),
                ("BOX", (2, 0), (2, 0), 1, SUCCESS_COLOR),
                ("BOX", (3, 0), (3, 0), 1, BRAND_COLOR),
                (
                    "BOX",
                    (4, 0),
                    (4, 0),
                    1,
                    DANGER_COLOR if high_risk > 0 else colors.gray,
                ),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ],
        ),
    )
    story.append(metrics_table)
    story.append(Spacer(1, 16))

    # Distribution bar
    story.append(Paragraph("Classification Distribution", styles["SubHeader"]))

    bar_width = 5.5 * inch
    pneumonia_bar_width = (pneumonia_pct / 100) * bar_width if successful > 0 else 0
    normal_bar_width = (normal_pct / 100) * bar_width if successful > 0 else 0

    dist_text = (
        f"<b>PNEUMONIA:</b> {pneumonia_count} ({pneumonia_pct:.1f}%)  |  "
        f"<b>NORMAL:</b> {normal_count} ({normal_pct:.1f}%)"
    )
    story.append(
        Paragraph(
            dist_text,
            ParagraphStyle("DistText", fontSize=9, alignment=TA_CENTER),
        ),
    )
    story.append(Spacer(1, 4))

    if successful > 0:
        bar_data = [
            [
                Paragraph(
                    "█" * int(pneumonia_pct / 2),
                    ParagraphStyle("PBar", fontSize=12, textColor=WARNING_COLOR),
                ),
                Paragraph(
                    "█" * int(normal_pct / 2),
                    ParagraphStyle("NBar", fontSize=12, textColor=SUCCESS_COLOR),
                ),
            ],
        ]
        bar_table = Table(bar_data, colWidths=[pneumonia_bar_width, normal_bar_width])
        story.append(bar_table)

    story.append(Spacer(1, 20))
