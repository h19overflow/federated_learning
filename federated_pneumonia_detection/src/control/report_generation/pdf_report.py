"""PDF Report Generation for Pneumonia Detection Results.

Generates professional clinical reports with X-ray images, heatmaps,
predictions, and AI-generated interpretations for doctor review.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)

# Color scheme matching the frontend
BRAND_COLOR = colors.HexColor("#267365")  # hsl(172, 63%, 28%)
SUCCESS_COLOR = colors.HexColor("#2D9E6E")  # green
WARNING_COLOR = colors.HexColor("#C4851A")  # amber
DANGER_COLOR = colors.HexColor("#B33030")  # red
LIGHT_BG = colors.HexColor("#F5FAF9")  # light teal background


def get_styles():
    """Get custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=BRAND_COLOR,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=BRAND_COLOR,
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=1,
        borderColor=BRAND_COLOR,
        borderPadding=4,
    ))

    styles.add(ParagraphStyle(
        name='SubHeader',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor("#1A4D45"),
        spaceBefore=10,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
    ))

    styles.add(ParagraphStyle(
        name='Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        leading=10,
        alignment=TA_JUSTIFY,
        spaceBefore=12,
    ))

    styles.add(ParagraphStyle(
        name='RiskHigh',
        parent=styles['Normal'],
        fontSize=12,
        textColor=DANGER_COLOR,
        fontName='Helvetica-Bold',
    ))

    styles.add(ParagraphStyle(
        name='RiskModerate',
        parent=styles['Normal'],
        fontSize=12,
        textColor=WARNING_COLOR,
        fontName='Helvetica-Bold',
    ))

    styles.add(ParagraphStyle(
        name='RiskLow',
        parent=styles['Normal'],
        fontSize=12,
        textColor=SUCCESS_COLOR,
        fontName='Helvetica-Bold',
    ))

    return styles


def base64_to_image(base64_string: str, max_width: float = 4*inch) -> Optional[RLImage]:
    """Convert base64 string to ReportLab Image."""
    try:
        image_data = base64.b64decode(base64_string)
        image_buffer = io.BytesIO(image_data)

        # Get image dimensions
        pil_image = Image.open(image_buffer)
        width, height = pil_image.size
        aspect_ratio = height / width

        # Reset buffer position
        image_buffer.seek(0)

        # Calculate dimensions maintaining aspect ratio
        img_width = min(max_width, 4*inch)
        img_height = img_width * aspect_ratio

        return RLImage(image_buffer, width=img_width, height=img_height)
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {e}")
        return None


def pil_to_reportlab_image(pil_image: Image.Image, max_width: float = 4*inch) -> Optional[RLImage]:
    """Convert PIL Image to ReportLab Image."""
    try:
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        width, height = pil_image.size
        aspect_ratio = height / width

        img_width = min(max_width, 4*inch)
        img_height = img_width * aspect_ratio

        return RLImage(buffer, width=img_width, height=img_height)
    except Exception as e:
        logger.error(f"Failed to convert PIL image: {e}")
        return None


def generate_prediction_report(
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
) -> bytes:
    """Generate a PDF report for a single prediction.

    Args:
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

    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    styles = get_styles()
    story = []

    # Header
    story.append(Paragraph("Pneumonia Detection Report", styles['ReportTitle']))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['ReportBody']
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_COLOR))
    story.append(Spacer(1, 12))

    # Study Information
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

    # Prediction Result
    story.append(Paragraph("AI Prediction Result", styles['SectionHeader']))

    is_pneumonia = prediction_class == "PNEUMONIA"
    result_color = WARNING_COLOR if is_pneumonia else SUCCESS_COLOR

    # Result box
    result_data = [
        [Paragraph(
            f"<b>Classification: {prediction_class}</b>",
            ParagraphStyle(
                'Result',
                fontSize=16,
                textColor=result_color,
                alignment=TA_CENTER,
            )
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

    # Probability breakdown
    story.append(Paragraph("Probability Breakdown", styles['SubHeader']))

    prob_data = [
        ["Class", "Probability", "Visual"],
        ["PNEUMONIA", f"{pneumonia_probability * 100:.1f}%", "█" * int(pneumonia_probability * 20)],
        ["NORMAL", f"{normal_probability * 100:.1f}%", "█" * int(normal_probability * 20)],
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
        ('TEXTCOLOR', (2, 1), (2, 1), WARNING_COLOR),  # Pneumonia bar
        ('TEXTCOLOR', (2, 2), (2, 2), SUCCESS_COLOR),  # Normal bar
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    # Images Section
    if original_image or heatmap_base64:
        story.append(Paragraph("Image Analysis", styles['SectionHeader']))

        images_row = []

        if original_image:
            orig_rl = pil_to_reportlab_image(original_image, max_width=2.8*inch)
            if orig_rl:
                images_row.append([
                    orig_rl,
                    Paragraph("Original X-Ray", ParagraphStyle('ImgCaption', fontSize=9, alignment=TA_CENTER))
                ])

        if heatmap_base64:
            heatmap_rl = base64_to_image(heatmap_base64, max_width=2.8*inch)
            if heatmap_rl:
                images_row.append([
                    heatmap_rl,
                    Paragraph("GradCAM Heatmap", ParagraphStyle('ImgCaption', fontSize=9, alignment=TA_CENTER))
                ])

        if images_row:
            if len(images_row) == 2:
                # Side by side
                img_table = Table([[images_row[0][0], images_row[1][0]]], colWidths=[3*inch, 3*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(img_table)

                caption_table = Table([[images_row[0][1], images_row[1][1]]], colWidths=[3*inch, 3*inch])
                caption_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(caption_table)
            else:
                # Single image
                story.append(images_row[0][0])
                story.append(images_row[0][1])

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "<i>Heatmap colors: Red/orange areas indicate regions that strongly "
            "influenced the model's prediction. Blue areas had less influence.</i>",
            styles['Disclaimer']
        ))
        story.append(Spacer(1, 16))

    # Clinical Interpretation
    if clinical_interpretation:
        story.append(Paragraph("Clinical Interpretation", styles['SectionHeader']))

        # Summary
        if clinical_interpretation.get('summary'):
            story.append(Paragraph("Summary", styles['SubHeader']))
            story.append(Paragraph(clinical_interpretation['summary'], styles['ReportBody']))
            story.append(Spacer(1, 8))

        # Confidence Explanation
        if clinical_interpretation.get('confidence_explanation'):
            story.append(Paragraph("Confidence Analysis", styles['SubHeader']))
            story.append(Paragraph(clinical_interpretation['confidence_explanation'], styles['ReportBody']))
            story.append(Spacer(1, 8))

        # Risk Assessment
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

        # Recommendations
        recommendations = clinical_interpretation.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("Recommendations", styles['SubHeader']))
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['ReportBody']))
            story.append(Spacer(1, 8))

    # Disclaimer
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

    # Footer with model info
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Model: {model_version} | Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')} | "
        "Powered by XRay Vision AI",
        ParagraphStyle('Footer', fontSize=7, textColor=colors.gray, alignment=TA_CENTER)
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _create_header(story, styles, title: str, subtitle: str = None):
    """Create a professional report header."""
    # Logo placeholder / Title area
    header_data = [[
        Paragraph(f"<b>XRAY VISION AI</b>", ParagraphStyle(
            'Logo', fontSize=10, textColor=BRAND_COLOR, fontName='Helvetica-Bold'
        )),
        Paragraph(title, styles['ReportTitle']),
        Paragraph(datetime.now().strftime('%Y-%m-%d'), ParagraphStyle(
            'Date', fontSize=10, textColor=colors.gray, alignment=TA_CENTER
        )),
    ]]
    header_table = Table(header_data, colWidths=[1.5*inch, 4*inch, 1.2*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
    ]))
    story.append(header_table)

    if subtitle:
        story.append(Paragraph(subtitle, ParagraphStyle(
            'Subtitle', fontSize=10, textColor=colors.gray, alignment=TA_CENTER
        )))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_COLOR))
    story.append(Spacer(1, 16))


def _create_executive_summary(story, styles, summary_stats: dict):
    """Create executive summary section with key metrics."""
    story.append(Paragraph("Executive Summary", styles['SectionHeader']))
    story.append(Spacer(1, 8))

    total = summary_stats.get('total_images', 0)
    successful = summary_stats.get('successful', 0)
    pneumonia_count = summary_stats.get('pneumonia_count', 0)
    normal_count = summary_stats.get('normal_count', 0)
    avg_confidence = summary_stats.get('avg_confidence', 0) * 100
    high_risk = summary_stats.get('high_risk_count', 0)

    # Key metrics in boxes
    pneumonia_pct = (pneumonia_count / max(successful, 1)) * 100
    normal_pct = (normal_count / max(successful, 1)) * 100

    # Create metric boxes
    def metric_box(label, value, color):
        return Table(
            [[Paragraph(f"<b>{value}</b>", ParagraphStyle('MetricValue', fontSize=18, textColor=color, alignment=TA_CENTER))],
             [Paragraph(label, ParagraphStyle('MetricLabel', fontSize=8, textColor=colors.gray, alignment=TA_CENTER))]],
            colWidths=[1.4*inch]
        )

    metrics_row = [
        metric_box("Total Analyzed", str(total), BRAND_COLOR),
        metric_box("Pneumonia", f"{pneumonia_count}", WARNING_COLOR),
        metric_box("Normal", f"{normal_count}", SUCCESS_COLOR),
        metric_box("Avg Confidence", f"{avg_confidence:.0f}%", BRAND_COLOR),
        metric_box("High Risk", str(high_risk), DANGER_COLOR if high_risk > 0 else colors.gray),
    ]

    metrics_table = Table([metrics_row], colWidths=[1.4*inch] * 5)
    metrics_table.setStyle(TableStyle([
        ('BOX', (0, 0), (0, 0), 1, BRAND_COLOR),
        ('BOX', (1, 0), (1, 0), 1, WARNING_COLOR),
        ('BOX', (2, 0), (2, 0), 1, SUCCESS_COLOR),
        ('BOX', (3, 0), (3, 0), 1, BRAND_COLOR),
        ('BOX', (4, 0), (4, 0), 1, DANGER_COLOR if high_risk > 0 else colors.gray),
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 16))

    # Distribution bar
    story.append(Paragraph("Classification Distribution", styles['SubHeader']))

    bar_width = 5.5 * inch
    pneumonia_bar_width = (pneumonia_pct / 100) * bar_width if successful > 0 else 0
    normal_bar_width = (normal_pct / 100) * bar_width if successful > 0 else 0

    dist_text = f"<b>PNEUMONIA:</b> {pneumonia_count} ({pneumonia_pct:.1f}%)  |  <b>NORMAL:</b> {normal_count} ({normal_pct:.1f}%)"
    story.append(Paragraph(dist_text, ParagraphStyle('DistText', fontSize=9, alignment=TA_CENTER)))
    story.append(Spacer(1, 4))

    # Visual bar
    if successful > 0:
        bar_data = [[
            Paragraph("█" * int(pneumonia_pct / 2), ParagraphStyle('PBar', fontSize=12, textColor=WARNING_COLOR)),
            Paragraph("█" * int(normal_pct / 2), ParagraphStyle('NBar', fontSize=12, textColor=SUCCESS_COLOR)),
        ]]
        bar_table = Table(bar_data, colWidths=[pneumonia_bar_width, normal_bar_width])
        story.append(bar_table)

    story.append(Spacer(1, 20))


def _create_results_table(story, styles, results: list):
    """Create detailed results table."""
    story.append(Paragraph("Detailed Results", styles['SectionHeader']))
    story.append(Spacer(1, 8))

    # Table header
    results_header = [
        Paragraph("<b>#</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Filename</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white)),
        Paragraph("<b>Classification</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Confidence</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Pneumonia %</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Normal %</b>", ParagraphStyle('TH', fontSize=8, textColor=colors.white, alignment=TA_CENTER)),
    ]
    results_data = [results_header]

    for i, result in enumerate(results[:100], 1):  # Limit to 100
        if result.get('success'):
            pred = result.get('prediction', {})
            pred_class = pred.get('predicted_class', 'N/A')
            conf = pred.get('confidence', 0) * 100
            pneumonia_prob = pred.get('pneumonia_probability', 0) * 100
            normal_prob = pred.get('normal_probability', 0) * 100

            # Color code the classification
            class_color = WARNING_COLOR if pred_class == 'PNEUMONIA' else SUCCESS_COLOR
            class_para = Paragraph(
                f"<b>{pred_class}</b>",
                ParagraphStyle('Class', fontSize=8, textColor=class_color, alignment=TA_CENTER)
            )
        else:
            pred_class = "ERROR"
            conf = 0
            pneumonia_prob = 0
            normal_prob = 0
            class_para = Paragraph(
                f"<b>ERROR</b>",
                ParagraphStyle('Class', fontSize=8, textColor=DANGER_COLOR, alignment=TA_CENTER)
            )

        filename = result.get('filename', 'Unknown')
        if len(filename) > 28:
            filename = filename[:25] + "..."

        results_data.append([
            Paragraph(str(i), ParagraphStyle('TD', fontSize=8, alignment=TA_CENTER)),
            Paragraph(filename, ParagraphStyle('TD', fontSize=8)),
            class_para,
            Paragraph(f"{conf:.1f}%", ParagraphStyle('TD', fontSize=8, alignment=TA_CENTER)),
            Paragraph(f"{pneumonia_prob:.1f}%", ParagraphStyle('TD', fontSize=8, alignment=TA_CENTER)),
            Paragraph(f"{normal_prob:.1f}%", ParagraphStyle('TD', fontSize=8, alignment=TA_CENTER)),
        ])

    results_table = Table(results_data, colWidths=[0.35*inch, 2.1*inch, 1.1*inch, 0.85*inch, 0.9*inch, 0.85*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_COLOR),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(results_table)

    if len(results) > 100:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<i>Note: Showing first 100 of {len(results)} results. Export to CSV for complete data.</i>",
            styles['Disclaimer']
        ))

    story.append(Spacer(1, 16))


def _create_appendix_page(story, styles, index: int, result: dict,
                          original_image: Optional[Image.Image] = None,
                          heatmap_base64: Optional[str] = None):
    """Create an appendix page for a single result with images."""
    story.append(PageBreak())

    filename = result.get('filename', 'Unknown')
    pred = result.get('prediction', {})
    pred_class = pred.get('predicted_class', 'N/A')
    confidence = pred.get('confidence', 0) * 100
    pneumonia_prob = pred.get('pneumonia_probability', 0) * 100
    normal_prob = pred.get('normal_probability', 0) * 100

    is_pneumonia = pred_class == 'PNEUMONIA'
    result_color = WARNING_COLOR if is_pneumonia else SUCCESS_COLOR

    # Appendix header
    header_text = f"Appendix {index}: {filename}"
    story.append(Paragraph(header_text, ParagraphStyle(
        'AppendixHeader', fontSize=14, textColor=BRAND_COLOR, fontName='Helvetica-Bold',
        spaceBefore=0, spaceAfter=8
    )))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_COLOR))
    story.append(Spacer(1, 12))

    # Result summary box
    result_box_data = [
        [Paragraph(f"<b>Classification: {pred_class}</b>", ParagraphStyle(
            'ResultClass', fontSize=14, textColor=result_color, alignment=TA_CENTER
        ))],
        [Paragraph(f"Confidence: {confidence:.1f}%", ParagraphStyle(
            'ResultConf', fontSize=11, alignment=TA_CENTER
        ))],
    ]
    result_box = Table(result_box_data, colWidths=[6*inch])
    result_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 2, result_color),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(result_box)
    story.append(Spacer(1, 12))

    # Probability breakdown
    prob_data = [
        [Paragraph("<b>Class</b>", ParagraphStyle('PH', fontSize=9, textColor=colors.white)),
         Paragraph("<b>Probability</b>", ParagraphStyle('PH', fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("<b>Indicator</b>", ParagraphStyle('PH', fontSize=9, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph("PNEUMONIA", ParagraphStyle('PD', fontSize=9)),
         Paragraph(f"{pneumonia_prob:.1f}%", ParagraphStyle('PD', fontSize=9, alignment=TA_CENTER)),
         Paragraph("●" * min(int(pneumonia_prob / 10), 10), ParagraphStyle('PD', fontSize=9, textColor=WARNING_COLOR, alignment=TA_CENTER))],
        [Paragraph("NORMAL", ParagraphStyle('PD', fontSize=9)),
         Paragraph(f"{normal_prob:.1f}%", ParagraphStyle('PD', fontSize=9, alignment=TA_CENTER)),
         Paragraph("●" * min(int(normal_prob / 10), 10), ParagraphStyle('PD', fontSize=9, textColor=SUCCESS_COLOR, alignment=TA_CENTER))],
    ]
    prob_table = Table(prob_data, colWidths=[1.5*inch, 1.2*inch, 3.3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    # Images section
    if original_image or heatmap_base64:
        story.append(Paragraph("Image Analysis", styles['SubHeader']))
        story.append(Spacer(1, 8))

        images_to_show = []
        captions = []

        if original_image:
            orig_rl = pil_to_reportlab_image(original_image, max_width=2.6*inch)
            if orig_rl:
                images_to_show.append(orig_rl)
                captions.append("Original X-Ray")

        if heatmap_base64:
            heatmap_rl = base64_to_image(heatmap_base64, max_width=2.6*inch)
            if heatmap_rl:
                images_to_show.append(heatmap_rl)
                captions.append("GradCAM Heatmap")

        if len(images_to_show) == 2:
            # Side by side with border
            img_table = Table([[images_to_show[0], Spacer(0.2*inch, 0), images_to_show[1]]],
                            colWidths=[2.8*inch, 0.2*inch, 2.8*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (0, 0), 1, colors.lightgrey),
                ('BOX', (2, 0), (2, 0), 1, colors.lightgrey),
            ]))
            story.append(img_table)

            # Captions
            caption_table = Table([
                [Paragraph(captions[0], ParagraphStyle('Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray)),
                 Spacer(0.2*inch, 0),
                 Paragraph(captions[1], ParagraphStyle('Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray))]
            ], colWidths=[2.8*inch, 0.2*inch, 2.8*inch])
            story.append(caption_table)
        elif len(images_to_show) == 1:
            story.append(images_to_show[0])
            story.append(Paragraph(captions[0], ParagraphStyle('Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray)))

        story.append(Spacer(1, 12))

        # Heatmap interpretation note
        if heatmap_base64:
            story.append(Paragraph(
                "<b>Heatmap Interpretation:</b> Warmer colors (red/orange) indicate regions that strongly "
                "influenced the model's prediction. These areas may show patterns associated with "
                f"{'pneumonia such as consolidation or infiltrates' if is_pneumonia else 'normal lung tissue'}.",
                ParagraphStyle('HeatmapNote', fontSize=8, textColor=colors.gray, leading=11)
            ))


def generate_batch_summary_report(
    results: list,
    summary_stats: dict,
    model_version: str = "unknown",
    images: Optional[list] = None,
    heatmaps: Optional[dict] = None,
) -> bytes:
    """Generate a comprehensive PDF report for batch predictions.

    Args:
        results: List of prediction result dicts
        summary_stats: Aggregate statistics dict
        model_version: Model version string
        images: Optional list of (filename, PIL.Image) tuples for appendix
        heatmaps: Optional dict mapping filename to heatmap base64 strings

    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18*mm,
        leftMargin=18*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
    )

    styles = get_styles()
    story = []

    # === PAGE 1: COVER & EXECUTIVE SUMMARY ===
    _create_header(story, styles, "Batch Analysis Report",
                   f"Comprehensive AI-Powered Pneumonia Detection Analysis")

    _create_executive_summary(story, styles, summary_stats)

    # === PAGE 2+: DETAILED RESULTS TABLE ===
    _create_results_table(story, styles, results)

    # === FINDINGS SUMMARY ===
    story.append(Paragraph("Key Findings", styles['SectionHeader']))
    story.append(Spacer(1, 8))

    total = summary_stats.get('total_images', 0)
    successful = summary_stats.get('successful', 0)
    pneumonia_count = summary_stats.get('pneumonia_count', 0)
    high_risk = summary_stats.get('high_risk_count', 0)
    avg_confidence = summary_stats.get('avg_confidence', 0) * 100

    findings = []
    if pneumonia_count > 0:
        findings.append(f"• <b>{pneumonia_count}</b> case(s) classified as <b>PNEUMONIA</b> requiring clinical review")
    if high_risk > 0:
        findings.append(f"• <b>{high_risk}</b> high-risk case(s) flagged for urgent attention")
    findings.append(f"• Average model confidence across all predictions: <b>{avg_confidence:.1f}%</b>")
    if successful < total:
        findings.append(f"• <b>{total - successful}</b> image(s) failed processing and require manual review")

    for finding in findings:
        story.append(Paragraph(finding, styles['ReportBody']))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 16))

    # === METHODOLOGY NOTE ===
    story.append(Paragraph("Methodology", styles['SubHeader']))
    story.append(Paragraph(
        f"This analysis was performed using the <b>{model_version}</b> deep learning model, "
        "a ResNet-50 architecture trained on chest X-ray images for binary classification "
        "(Normal vs Pneumonia). The model uses GradCAM (Gradient-weighted Class Activation Mapping) "
        "to generate interpretable heatmaps highlighting regions influencing each prediction.",
        styles['ReportBody']
    ))
    story.append(Spacer(1, 16))

    # === DISCLAIMER ===
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>IMPORTANT MEDICAL DISCLAIMER</b>",
        ParagraphStyle('DisclaimerTitle', fontSize=10, textColor=DANGER_COLOR, fontName='Helvetica-Bold')
    ))
    story.append(Paragraph(
        "This report is generated by an AI-assisted diagnostic system and is intended for "
        "informational and screening purposes only. It should NOT be used as the sole basis "
        "for clinical decisions or diagnoses. All findings must be reviewed and validated by "
        "a qualified radiologist or physician. AI systems have limitations and can produce "
        "false positives and false negatives. Clinical judgment should always take precedence. "
        "This system is designed to assist, not replace, medical professionals.",
        styles['Disclaimer']
    ))

    # === FOOTER ===
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    footer_data = [[
        Paragraph("XRay Vision AI", ParagraphStyle('FootL', fontSize=7, textColor=colors.gray)),
        Paragraph(f"Model: {model_version}", ParagraphStyle('FootC', fontSize=7, textColor=colors.gray, alignment=TA_CENTER)),
        Paragraph(f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}", ParagraphStyle('FootR', fontSize=7, textColor=colors.gray, alignment=TA_CENTER)),
    ]]
    footer_table = Table(footer_data, colWidths=[2*inch, 2.5*inch, 2*inch])
    story.append(footer_table)

    # === APPENDIX: INDIVIDUAL IMAGE ANALYSIS ===
    if images and len(images) > 0:
        # Create appendix for each image (limit to first 20 for PDF size)
        images_dict = {filename: img for filename, img in images}
        heatmaps = heatmaps or {}

        appendix_count = 0
        for i, result in enumerate(results):
            if appendix_count >= 20:
                break

            filename = result.get('filename', '')
            if filename in images_dict and result.get('success'):
                appendix_count += 1
                _create_appendix_page(
                    story, styles, appendix_count, result,
                    original_image=images_dict.get(filename),
                    heatmap_base64=heatmaps.get(filename)
                )

        if len(results) > 20:
            story.append(PageBreak())
            story.append(Paragraph(
                f"Note: Appendix limited to first 20 images. "
                f"Total images in batch: {len(results)}",
                styles['Disclaimer']
            ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
