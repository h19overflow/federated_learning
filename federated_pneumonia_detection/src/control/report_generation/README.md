# Report Generation Submodule

## Purpose
The Report Generation submodule provides functionality to generate professional, clinical-grade PDF reports for pneumonia detection results.

## Problem Solved
Raw AI predictions are difficult for clinicians to interpret in isolation. This submodule converts model outputs, images, heatmaps, and clinical interpretations into a structured, printable PDF format that follows medical reporting standards.

## How It Works
The submodule uses the **ReportLab** library to programmatically build PDF documents. It follows a modular approach where different sections of the report (Header, Study Info, Results, Images, Interpretation) are built by specialized functions and then assembled into a final "story".

## Key Files
- `pdf_report.py` - Main facade providing `generate_prediction_report` and `generate_batch_summary_report`.
- `internals/sections/` - Modular builders for different report sections.
- `internals/styles.py` - Centralized ReportLab styles and branding colors.
- `internals/images/` - Utilities for embedding PIL images and Base64 heatmaps into PDFs.

## Dependencies
- Requires: `reportlab`, `PIL`.
- Used by: `api/endpoints/inference/`

## Architecture
```mermaid
graph TD
    PR[pdf_report.py] --> BS[build_single_report]
    PR --> BB[build_batch_report]
    
    BS --> H[Header]
    BS --> SI[Study Info]
    BS --> PR[Prediction Result]
    BS --> IS[Images Section]
    BS --> CI[Clinical Interpretation]
    
    BB --> ES[Executive Summary]
    BB --> RT[Results Table]
    BB --> AP[Appendix]
```

## Integration Points
- **Upstream**: Called by FastAPI endpoints to provide downloadable reports.
- **Downstream**: Consumes data from `InferenceService` and `ClinicalInterpreter`.
