"""Appendix page builder for individual image analysis."""

from typing import Optional

from PIL import Image
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
    TableStyle,
)

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


def create_appendix_page(
    story: list,
    styles,
    index: int,
    result: dict,
    original_image: Optional[Image.Image] = None,
    heatmap_base64: Optional[str] = None,
) -> None:
    """Create an appendix page for a single result with images.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        index: Appendix index number
        result: Prediction result dictionary
        original_image: Original X-ray PIL Image
        heatmap_base64: GradCAM heatmap as base64 string
    """
    story.append(PageBreak())

    filename = result.get('filename', 'Unknown')
    pred = result.get('prediction', {})
    pred_class = pred.get('predicted_class', 'N/A')
    confidence = pred.get('confidence', 0) * 100
    pneumonia_prob = pred.get('pneumonia_probability', 0) * 100
    normal_prob = pred.get('normal_probability', 0) * 100

    is_pneumonia = pred_class == 'PNEUMONIA'
    result_color = WARNING_COLOR if is_pneumonia else SUCCESS_COLOR

    _add_appendix_header(story, index, filename)
    _add_result_summary_box(story, pred_class, confidence, result_color)
    _add_probability_breakdown(story, pneumonia_prob, normal_prob)
    _add_images_section(story, styles, original_image, heatmap_base64, is_pneumonia)


def _add_appendix_header(story: list, index: int, filename: str) -> None:
    """Add appendix header with title."""
    header_text = f"Appendix {index}: {filename}"
    story.append(Paragraph(header_text, ParagraphStyle(
        'AppendixHeader', fontSize=14, textColor=BRAND_COLOR, fontName='Helvetica-Bold',
        spaceBefore=0, spaceAfter=8
    )))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_COLOR))
    story.append(Spacer(1, 12))


def _add_result_summary_box(
    story: list, pred_class: str, confidence: float, result_color
) -> None:
    """Add classification result summary box."""
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


def _add_probability_breakdown(
    story: list, pneumonia_prob: float, normal_prob: float
) -> None:
    """Add probability breakdown table."""
    prob_data = [
        [Paragraph("<b>Class</b>", ParagraphStyle(
            'PH', fontSize=9, textColor=colors.white
        )),
         Paragraph("<b>Probability</b>", ParagraphStyle(
             'PH', fontSize=9, textColor=colors.white, alignment=TA_CENTER
         )),
         Paragraph("<b>Indicator</b>", ParagraphStyle(
             'PH', fontSize=9, textColor=colors.white, alignment=TA_CENTER
         ))],
        [Paragraph("PNEUMONIA", ParagraphStyle('PD', fontSize=9)),
         Paragraph(f"{pneumonia_prob:.1f}%", ParagraphStyle(
             'PD', fontSize=9, alignment=TA_CENTER
         )),
         Paragraph("●" * min(int(pneumonia_prob / 10), 10), ParagraphStyle(
             'PD', fontSize=9, textColor=WARNING_COLOR, alignment=TA_CENTER
         ))],
        [Paragraph("NORMAL", ParagraphStyle('PD', fontSize=9)),
         Paragraph(f"{normal_prob:.1f}%", ParagraphStyle(
             'PD', fontSize=9, alignment=TA_CENTER
         )),
         Paragraph("●" * min(int(normal_prob / 10), 10), ParagraphStyle(
             'PD', fontSize=9, textColor=SUCCESS_COLOR, alignment=TA_CENTER
         ))],
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


def _add_images_section(
    story: list,
    styles,
    original_image: Optional[Image.Image],
    heatmap_base64: Optional[str],
    is_pneumonia: bool,
) -> None:
    """Add images section with X-ray and heatmap."""
    if not original_image and not heatmap_base64:
        return

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
        img_table = Table(
            [[images_to_show[0], Spacer(0.2*inch, 0), images_to_show[1]]],
            colWidths=[2.8*inch, 0.2*inch, 2.8*inch]
        )
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (0, 0), 1, colors.lightgrey),
            ('BOX', (2, 0), (2, 0), 1, colors.lightgrey),
        ]))
        story.append(img_table)

        caption_table = Table([
            [Paragraph(captions[0], ParagraphStyle(
                'Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray
            )),
             Spacer(0.2*inch, 0),
             Paragraph(captions[1], ParagraphStyle(
                 'Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray
             ))]
        ], colWidths=[2.8*inch, 0.2*inch, 2.8*inch])
        story.append(caption_table)
    elif len(images_to_show) == 1:
        story.append(images_to_show[0])
        story.append(Paragraph(captions[0], ParagraphStyle(
            'Cap', fontSize=9, alignment=TA_CENTER, textColor=colors.gray
        )))

    story.append(Spacer(1, 12))

    if heatmap_base64:
        pattern_desc = (
            "pneumonia such as consolidation or infiltrates"
            if is_pneumonia else "normal lung tissue"
        )
        story.append(Paragraph(
            "<b>Heatmap Interpretation:</b> Warmer colors (red/orange) indicate "
            "regions that strongly influenced the model's prediction. These areas "
            f"may show patterns associated with {pattern_desc}.",
            ParagraphStyle('HeatmapNote', fontSize=8, textColor=colors.gray, leading=11)
        ))
