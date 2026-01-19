"""Report section builders for PDF generation."""

from federated_pneumonia_detection.src.control.report_generation.internals.sections.appendix import (
    create_appendix_page,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.batch_report import (
    build_batch_report,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.executive_summary import (
    create_executive_summary,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.header import (
    create_header,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.results_table import (
    create_results_table,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.single_report import (
    build_single_report,
)

__all__ = [
    "create_header",
    "create_executive_summary",
    "create_results_table",
    "create_appendix_page",
    "build_single_report",
    "build_batch_report",
]
