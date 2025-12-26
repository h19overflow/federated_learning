# Initial Concept
OK so the main thing is this is just a project for an FIP Final year project, requiring a defined scope, measurable objectives, technical specifications, and a clear implementation roadmap with validation criteria.

# Product Guide

## Product Vision
A sophisticated, privacy-preserving federated learning system designed to detect pneumonia from chest X-ray images. This Final Year Project (FYP) aims to demonstrate the viability of collaborative AI training across multiple medical institutions without sharing sensitive patient data, satisfying rigorous academic and technical standards.

## Target Audience
- **Primary:** Academic Examiners and Reviewers assessing the project's technical depth, architectural soundness, and research validity.
- **Secondary:** Medical Researchers and Healthcare Institutions interested in privacy-preserving collaborative AI.
- **Tertiary:** Future students or developers looking for a reference implementation of a robust federated learning system.

## Key Goals & Objectives
1.  **Academic Rigor:** Deliver a project with a clearly defined scope, measurable objectives, and a comprehensive validation methodology suitable for a Final Year Project.
2.  **Comparative Analysis:** Conduct and present a rigorous comparison between Centralized and Federated Learning models, using metrics like Accuracy, Sensitivity, Specificity, and AUC-ROC.
3.  **Privacy-Preserving Architecture:** Implement and validate a federated learning framework (using Flower) that ensures patient data remains local and private.
4.  **Engineering Excellence:** Demonstrate high-quality software engineering practices, including a clean Entity-Control-Boundary (ECB) architecture, comprehensive testing (unit, integration), and a modern tech stack.
5.  **User Experience:** Provide an intuitive interface for configuring experiments, monitoring training progress in real-time, and visualizing comparative results.

## Core Features
-   **Dual Training Modes:** Support for both Centralized (baseline) and Federated Learning training workflows.
-   **Advanced Experiment Orchestration:** Tools to configure, run, and manage complex training experiments with varying parameters.
-   **Flexible Data Partitioning:** Support for IID, Non-IID (Patient-based), and Stratified data splitting to simulate real-world scenarios.
-   **Real-time Visualization:** WebSocket-powered dashboards for monitoring training metrics (loss, accuracy) and system status.
-   **Research Assistant Integration:** An AI-powered assistant (using RAG and Arxiv) to help users explore relevant literature and interpret results.
-   **Comprehensive Evaluation:** Automated generation of confusion matrices, ROC curves, and detailed performance reports.
