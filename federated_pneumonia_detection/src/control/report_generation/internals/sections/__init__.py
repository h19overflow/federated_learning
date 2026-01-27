"""Report section builders for PDF generation."""

from federated_pneumonia_detection.src.control.report_generation.internals.sections.appendix import (  # noqa: E501
    create_appendix_page,
)
from federated_pneumonia_detection.src.control.report_generation.internals.sections.batch_report import (  # noqa: E501
    build_batch_report,
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
from federated_pneumonia_detection.src.control.report_generation.internals.sections.single_report import (  # noqa: E501
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
