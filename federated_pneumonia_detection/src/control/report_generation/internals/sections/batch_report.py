"""Batch prediction report builder."""

from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
)

from federated_pneumonia_detection.src.control.report_generation.internals.constants import (  # noqa: E501
    DANGER_COLOR,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.appendix import (  # noqa: E501
    create_appendix_page,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.executive_summary import (  # noqa: E501
    create_executive_summary,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.header import (  # noqa: E501
    create_header,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.results_table import (  # noqa: E501
    create_results_table,
)


def build_batch_report(
    story: list,
    styles,
    results: list,
    summary_stats: dict,
    model_version: str = "unknown",
    images: Optional[list] = None,
    heatmaps: Optional[dict] = None,
) -> None:
    """Build all sections for a batch prediction report.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        results: List of prediction result dictionaries
        summary_stats: Aggregate statistics dictionary
        model_version: Model version string
        images: Optional list of (filename, PIL.Image) tuples for appendix
        heatmaps: Optional dict mapping filename to heatmap base64 strings
    """
    # Cover & Executive Summary
    create_header(
        story,
        styles,
        "Batch Analysis Report",
        "Comprehensive AI-Powered Pneumonia Detection Analysis",
    )
    create_executive_summary(story, styles, summary_stats)

    # Detailed Results Table
    create_results_table(story, styles, results)

    # Key Findings
    _add_key_findings(story, styles, summary_stats)

    # Methodology
    _add_methodology(story, styles, model_version)

    # Disclaimer
    _add_disclaimer(story, styles)

    # Footer
    _add_footer(story, model_version)

    # Appendix with individual images
    _add_appendix(story, styles, results, images, heatmaps)


def _add_key_findings(story: list, styles, summary_stats: dict) -> None:
    """Add key findings section."""
    story.append(Paragraph("Key Findings", styles["SectionHeader"]))
    story.append(Spacer(1, 8))

    total = summary_stats.get("total_images", 0)
    successful = summary_stats.get("successful", 0)
    pneumonia_count = summary_stats.get("pneumonia_count", 0)
    high_risk = summary_stats.get("high_risk_count", 0)
    avg_confidence = summary_stats.get("avg_confidence", 0) * 100

    findings = []
    if pneumonia_count > 0:
        findings.append(
            f"• <b>{pneumonia_count}</b> case(s) classified as "
            "<b>PNEUMONIA</b> requiring clinical review",
        )
    if high_risk > 0:
        findings.append(
            f"• <b>{high_risk}</b> high-risk case(s) flagged for urgent attention",
        )
    findings.append(
        f"• Average model confidence across all predictions: "
        f"<b>{avg_confidence:.1f}%</b>",
    )
    if successful < total:
        findings.append(
            f"• <b>{total - successful}</b> image(s) failed processing "
            "and require manual review",
        )

    for finding in findings:
        story.append(Paragraph(finding, styles["ReportBody"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 16))


def _add_methodology(story: list, styles, model_version: str) -> None:
    """Add methodology section."""
    story.append(Paragraph("Methodology", styles["SubHeader"]))
    story.append(
        Paragraph(
            f"This analysis was performed using the <b>{model_version}</b> "
            "deep learning model, a ResNet-50 architecture trained on chest X-ray "
            "images for binary classification (Normal vs Pneumonia). The model uses "
            "GradCAM (Gradient-weighted Class Activation Mapping) to generate "
            "interpretable heatmaps highlighting regions influencing each prediction.",
            styles["ReportBody"],
        ),
    )
    story.append(Spacer(1, 16))


def _add_disclaimer(story: list, styles) -> None:
    """Add medical disclaimer section."""
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>IMPORTANT MEDICAL DISCLAIMER</b>",
            ParagraphStyle(
                "DisclaimerTitle",
                fontSize=10,
                textColor=DANGER_COLOR,
                fontName="Helvetica-Bold",
            ),
        ),
    )
    story.append(
        Paragraph(
            "This report is generated by an AI-assisted diagnostic system and is "
            "intended for informational and screening purposes only. It should NOT "
            "be used as the sole basis for clinical decisions or diagnoses. "
            "All findings must be reviewed and validated by a qualified radiologist "
            "or physician. AI systems have limitations and can produce false "
            "positives and false negatives. Clinical judgment should always take "
            "precedence. This system is designed to assist, not replace, medical "
            "professionals.",
            styles["Disclaimer"],
        ),
    )


def _add_footer(story: list, model_version: str) -> None:
    """Add report footer."""
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    footer_data = [
        [
            Paragraph(
                "XRay Vision AI",
                ParagraphStyle("FootL", fontSize=7, textColor=colors.gray),
            ),
            Paragraph(
                f"Model: {model_version}",
                ParagraphStyle(
                    "FootC",
                    fontSize=7,
                    textColor=colors.gray,
                    alignment=TA_CENTER,
                ),
            ),
            Paragraph(
                f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}",
                ParagraphStyle(
                    "FootR",
                    fontSize=7,
                    textColor=colors.gray,
                    alignment=TA_CENTER,
                ),
            ),
        ],
    ]
    footer_table = Table(footer_data, colWidths=[2 * inch, 2.5 * inch, 2 * inch])
    story.append(footer_table)


def _add_appendix(
    story: list,
    styles,
    results: list,
    images: Optional[list],
    heatmaps: Optional[dict],
) -> None:
    """Add appendix with individual image analysis pages."""
    if not images or len(images) == 0:
        return

    images_dict = {filename: img for filename, img in images}
    heatmaps = heatmaps or {}

    appendix_count = 0
    for result in results:
        if appendix_count >= 20:
            break

        filename = result.get("filename", "")
        if filename in images_dict and result.get("success"):
            appendix_count += 1
            create_appendix_page(
                story,
                styles,
                appendix_count,
                result,
                original_image=images_dict.get(filename),
                heatmap_base64=heatmaps.get(filename),
            )

    if len(results) > 20:
        story.append(PageBreak())
        story.append(
            Paragraph(
                f"Note: Appendix limited to first 20 images. "
                f"Total images in batch: {len(results)}",
                styles["Disclaimer"],
            ),
        )
