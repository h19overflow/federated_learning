"""Single prediction report builder."""

from datetime import datetime
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, Spacer, Table, TableStyle

from federated_pneumonia_detection.src.control.report_generation.utils.constants import (
    BRAND_COLOR,
    LIGHT_BG,
    SUCCESS_COLOR,
    WARNING_COLOR,
)
from federated_pneumonia_detection.src.control.report_generation.utils.images import (
    base64_to_image,
    pil_to_reportlab_image,
)


def build_single_report(
    story: list,
    styles,
    prediction_class: str,
    confidence: float,
    pneumonia_probability: float,
    normal_probability: float,
    original_image: Optional[Image.Image] = None,
    heatmap_base64: Optional[str] = None,
    clinical_interpretation: Optional[dict] = None,
    filename: Optional[str] = None,
    model_version: str = "unknown",
    processing_time_ms: float = 0.0,
) -> None:
    """Build all sections for a single prediction report.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        prediction_class: NORMAL or PNEUMONIA
        confidence: Model confidence (0-1)
        pneumonia_probability: Probability of pneumonia (0-1)
        normal_probability: Probability of normal (0-1)
        original_image: Original X-ray image (PIL Image)
        heatmap_base64: GradCAM heatmap as base64 string
        clinical_interpretation: Dict with summary, risk_assessment, recommendations
        filename: Original filename
        model_version: Model version string
        processing_time_ms: Processing time in milliseconds
    """
    _add_header(story, styles)
    _add_study_info(story, styles, filename, model_version, processing_time_ms)
    _add_prediction_result(story, styles, prediction_class, confidence)
    _add_probability_breakdown(
        story, styles, pneumonia_probability, normal_probability
    )
    _add_images_section(story, styles, original_image, heatmap_base64)
    _add_clinical_interpretation(story, styles, clinical_interpretation)
    _add_disclaimer(story, styles)
    _add_footer(story, model_version)


def _add_header(story: list, styles) -> None:
    """Add report header."""
    story.append(Paragraph("Pneumonia Detection Report", styles['ReportTitle']))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['ReportBody']
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_COLOR))
    story.append(Spacer(1, 12))


def _add_study_info(
    story: list, styles, filename: Optional[str],
    model_version: str, processing_time_ms: float
) -> None:
    """Add study information section."""
    story.append(Paragraph("Study Information", styles['SectionHeader']))

    study_info = [
        ["Filename:", filename or "Unknown"],
        ["Analysis Date:", datetime.now().strftime('%Y-%m-%d')],
        ["Model Version:", model_version],
        ["Processing Time:", f"{processing_time_ms:.0f} ms"],
    ]

    study_table = Table(study_info, colWidths=[2*inch, 4*inch])
    study_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), BRAND_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(study_table)
    story.append(Spacer(1, 16))


def _add_prediction_result(
    story: list, styles, prediction_class: str, confidence: float
) -> None:
    """Add prediction result section."""
    story.append(Paragraph("AI Prediction Result", styles['SectionHeader']))

    is_pneumonia = prediction_class == "PNEUMONIA"
    result_color = WARNING_COLOR if is_pneumonia else SUCCESS_COLOR

    result_data = [
        [Paragraph(
            f"<b>Classification: {prediction_class}</b>",
            ParagraphStyle('Result', fontSize=16, textColor=result_color, alignment=TA_CENTER)
        )],
        [Paragraph(
            f"Confidence: {confidence * 100:.1f}%",
            ParagraphStyle('Conf', fontSize=12, alignment=TA_CENTER)
        )],
    ]

    result_table = Table(result_data, colWidths=[5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 2, result_color),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 12))


def _add_probability_breakdown(
    story: list, styles, pneumonia_probability: float, normal_probability: float
) -> None:
    """Add probability breakdown section."""
    story.append(Paragraph("Probability Breakdown", styles['SubHeader']))

    prob_data = [
        ["Class", "Probability", "Visual"],
        ["PNEUMONIA", f"{pneumonia_probability * 100:.1f}%",
         "█" * int(pneumonia_probability * 20)],
        ["NORMAL", f"{normal_probability * 100:.1f}%",
         "█" * int(normal_probability * 20)],
    ]

    prob_table = Table(prob_data, colWidths=[1.5*inch, 1.2*inch, 3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (2, 1), (2, 1), WARNING_COLOR),
        ('TEXTCOLOR', (2, 2), (2, 2), SUCCESS_COLOR),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))


def _add_images_section(
    story: list, styles,
    original_image: Optional[Image.Image],
    heatmap_base64: Optional[str]
) -> None:
    """Add images section with X-ray and heatmap."""
    if not original_image and not heatmap_base64:
        return

    story.append(Paragraph("Image Analysis", styles['SectionHeader']))

    images_row = []

    if original_image:
        orig_rl = pil_to_reportlab_image(original_image, max_width=2.8*inch)
        if orig_rl:
            images_row.append([
                orig_rl,
                Paragraph("Original X-Ray", ParagraphStyle(
                    'ImgCaption', fontSize=9, alignment=TA_CENTER
                ))
            ])

    if heatmap_base64:
        heatmap_rl = base64_to_image(heatmap_base64, max_width=2.8*inch)
        if heatmap_rl:
            images_row.append([
                heatmap_rl,
                Paragraph("GradCAM Heatmap", ParagraphStyle(
                    'ImgCaption', fontSize=9, alignment=TA_CENTER
                ))
            ])

    if images_row:
        if len(images_row) == 2:
            img_table = Table(
                [[images_row[0][0], images_row[1][0]]],
                colWidths=[3*inch, 3*inch]
            )
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(img_table)

            caption_table = Table(
                [[images_row[0][1], images_row[1][1]]],
                colWidths=[3*inch, 3*inch]
            )
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(caption_table)
        else:
            story.append(images_row[0][0])
            story.append(images_row[0][1])

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Heatmap colors: Red/orange areas indicate regions that strongly "
        "influenced the model's prediction. Blue areas had less influence.</i>",
        styles['Disclaimer']
    ))
    story.append(Spacer(1, 16))


def _add_clinical_interpretation(
    story: list, styles, clinical_interpretation: Optional[dict]
) -> None:
    """Add clinical interpretation section."""
    if not clinical_interpretation:
        return

    story.append(Paragraph("Clinical Interpretation", styles['SectionHeader']))

    if clinical_interpretation.get('summary'):
        story.append(Paragraph("Summary", styles['SubHeader']))
        story.append(Paragraph(clinical_interpretation['summary'], styles['ReportBody']))
        story.append(Spacer(1, 8))

    if clinical_interpretation.get('confidence_explanation'):
        story.append(Paragraph("Confidence Analysis", styles['SubHeader']))
        story.append(Paragraph(
            clinical_interpretation['confidence_explanation'], styles['ReportBody']
        ))
        story.append(Spacer(1, 8))

    risk = clinical_interpretation.get('risk_assessment', {})
    if risk:
        story.append(Paragraph("Risk Assessment", styles['SubHeader']))

        risk_level = risk.get('risk_level', 'UNKNOWN')
        risk_style = 'RiskHigh' if risk_level in ['HIGH', 'CRITICAL'] else (
            'RiskModerate' if risk_level == 'MODERATE' else 'RiskLow'
        )
        story.append(Paragraph(f"Risk Level: {risk_level}", styles[risk_style]))

        if risk.get('false_negative_risk'):
            story.append(Paragraph(
                f"False Negative Risk: {risk['false_negative_risk']}",
                styles['ReportBody']
            ))

        if risk.get('factors'):
            story.append(Paragraph("Contributing Factors:", styles['ReportBody']))
            for factor in risk['factors']:
                story.append(Paragraph(f"  • {factor}", styles['ReportBody']))

        story.append(Spacer(1, 8))

    recommendations = clinical_interpretation.get('recommendations', [])
    if recommendations:
        story.append(Paragraph("Recommendations", styles['SubHeader']))
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['ReportBody']))
        story.append(Spacer(1, 8))


def _add_disclaimer(story: list, styles) -> None:
    """Add disclaimer section."""
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI-assisted "
        "diagnostic system and is intended for informational purposes only. It should "
        "NOT be used as the sole basis for clinical decisions. The predictions and "
        "interpretations provided must be reviewed and validated by a qualified "
        "radiologist or physician. AI systems can make errors, and clinical judgment "
        "should always take precedence. This system is designed to assist, not replace, "
        "medical professionals.",
        styles['Disclaimer']
    ))


def _add_footer(story: list, model_version: str) -> None:
    """Add footer with model info."""
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Model: {model_version} | Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')} | "
        "Powered by XRay Vision AI",
        ParagraphStyle('Footer', fontSize=7, textColor=colors.gray, alignment=TA_CENTER)
    ))
