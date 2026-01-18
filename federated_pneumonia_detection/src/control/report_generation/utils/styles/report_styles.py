"""Custom paragraph styles for PDF reports."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from federated_pneumonia_detection.src.control.report_generation.utils.constants import (
    BRAND_COLOR,
    DANGER_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
)


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
