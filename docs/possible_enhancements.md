# Enhancement Plan: FYP2 Pneumonia Detection System

**Date:** January 11, 2026  
**Status:** Proposed  
**Context:** Hybrid Federated/Centralized Pneumonia Detection using ResNet & Flower

## 1. Executive Summary
This document outlines a strategic roadmap for enhancing the FYP2 system, moving from a research prototype to a clinical-grade platform. The plan synthesizes architectural innovations (agents, privacy, clinical integration) with robust backend engineering (scalability, security, observability).

---

## 2. System & Architectural Innovations (High Value)

### 🤖 Agentic Clinical Workflows
Leverage the existing chat architecture to add specialized medical agents.
- **Automated Clinical Report Agent**: Generates structured patient reports from inference results, highlighting false negative risks and confidence scores.
- **Experiment Interpretation Assistant**: LLM-driven analysis of training dynamics, comparing centralized vs. federated runs and suggesting hyperparameter optimizations.

### 🔒 Privacy-Preserving Federated Learning
Enhance the Flower-based engine for real-world deployment.
- **Differential Privacy (DP)**: Implement DP-SGD with calibrated noise to guarantee privacy for client updates.
- **Secure Aggregation**: Use Homomorphic Encryption or SMPC to blind the server to individual hospital updates.
- **Personalized FL**: Implement FedProx or local fine-tuning to adapt the global model to site-specific patient demographics.

### 🏥 Clinical Integration
Bridge the gap between ML research and hospital infrastructure.
- **DICOM / PACS Integration**: Direct ingestion of `.dcm` files from hospital PACS servers, including automated de-identification.
- **FHIR Interoperability**: Export predictions and patient observations in HL7 FHIR standard for EHR integration.
- **Multi-Modal Training**: Upgrade ResNet to a multi-input architecture fusing X-ray imagery with tabular clinical metadata (age, comorbidities).

---

## 3. Backend & API Enhancements (Engineering)

### ⚡ Experiment Management
Improve control over long-running training jobs.
- **Pause/Resume**: Task queue (Celery/Redis) integration to suspend and resume training runs.
- **Graceful Cancellation**: Robust resource cleanup (GPU memory, temp files) when users abort experiments.
- **Validation-Only Mode**: Run evaluation on existing checkpoints without retraining.

### 📊 Advanced Analytics & Observability
Deepen insights into model behavior.
- **Drift Detection API**: Monitor training vs. inference data distributions (KL divergence) to alert on model degradation.
- **Explainability Dashboard**: Integration of Grad-CAM heatmaps to visualize pneumonia localization on X-rays.
- **Temporal Trends**: Aggregated metrics over time to detect seasonal or operational shifts.

### 🛡️ Security & Performance
Harden the API for multi-tenant usage.
- **Authentication**: JWT-based auth with role-based access control (Admin/Researcher/Viewer).
- **Rate Limiting**: Throttling on training endpoints to prevent DoS.
- **Response Caching**: Redis-backed caching for heavy analytics endpoints (`/api/runs/analytics`).
- **Input Sanitization**: Validation against injection attacks and malicious file uploads.

### 💾 Data & Export
- **Batch Export**: Bulk download of run results and logs.
- **Model Artifact Export**: Standardized export of trained weights (`.pth`, `.onnx`) with architecture metadata.

---

## 4. Developer Experience & Quality

- **API Documentation**: Enhanced OpenAPI schemas with examples and response models.
- **Testing Strategy**: Load testing scripts (Locust) and contract tests for API stability.
- **Config Management**: Validation endpoints for training configs, versioned history, and diff tools.
- **Code Quality**: Strict typing, deduplication of utility functions, and centralized error handling.

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- **Focus**: Stability & Observability
- Implement `Pause/Resume` and `Cancellation`.
- Add `Response Caching` and `Centralized Error Handling`.
- Deploy `Explainability Dashboard` (Grad-CAM).

### Phase 2: Intelligence (Weeks 4-8)
- **Focus**: Agentic Capabilities
- Build `Clinical Report Agent` and `Interpretation Assistant`.
- Implement `Drift Detection` analytics.

### Phase 3: Privacy & Integration (Months 3-4)
- **Focus**: Clinical Deployment
- Roll out `Differential Privacy` for Federated Learning.
- Build `DICOM` ingestion pipeline.
- Implement `FHIR` export.

### Phase 4: Advanced Research (Months 5+)
- **Focus**: SOTA Improvements
- Develop `Personalized FL` strategies.
- Experiment with `Multi-Modal` architectures.
