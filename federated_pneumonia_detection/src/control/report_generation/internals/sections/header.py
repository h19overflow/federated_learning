"""Report header section builder."""

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, Spacer, Table, TableStyle

from federated_pneumonia_detection.src.control.report_generation.internals.constants import (
    BRAND_COLOR,
)


def create_header(story: list, styles, title: str, subtitle: str = None) -> None:
    """Create a professional report header.

    Args:
        story: ReportLab story list to append elements to
        styles: Paragraph styles dictionary
        title: Main title for the report
        subtitle: Optional subtitle text
    """
    header_data = [[
        Paragraph("<b>XRAY VISION AI</b>", ParagraphStyle(
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
