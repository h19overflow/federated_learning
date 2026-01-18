# Notes: Component Structure Analysis

## Current Structure

### Root Level Components (src/components/)
1. AnalyticsTab.tsx
2. ChatSidebar.tsx
3. DatasetUpload.tsx
4. ExperimentConfig.tsx
5. Footer.tsx
6. Header.tsx
7. HelpTooltip.tsx
8. InstructionCard.tsx
9. KnowledgeBasePanel.tsx
10. LoadingOverlay.tsx
11. MetadataDisplay.tsx
12. ProgressIndicator.tsx
13. ResultsVisualization.tsx
14. StepContent.tsx
15. StepIndicator.tsx
16. TrainingExecution.tsx
17. WelcomeGuide.tsx

### Organized Subdirectories

#### inference/ (8 files + index.ts)
- BatchExportButton.tsx
- BatchResultsGrid.tsx
- BatchSummaryStats.tsx
- BatchUploadZone.tsx
- ClinicalInterpretation.tsx
- HeatmapOverlay.tsx
- ImageDropzone.tsx
- InferenceStatusBadge.tsx
- PredictionResult.tsx
- ResultDetailModal.tsx
- index.ts (barrel export)

#### observability/ (2 files + index.ts)
- BatchMetricsChart.tsx
- TrainingObservabilityPanel.tsx
- index.ts (barrel export)

#### ui/ (45 files)
- All shadcn/ui primitive components
- Shared UI components like button, card, dialog, etc.

## Component Analysis (by functionality)

### Training Workflow Components
- DatasetUpload.tsx - Dataset upload, validation, train/val split configuration
- ExperimentConfig.tsx - Training parameters (centralized/federated mode, hyperparams)
- TrainingExecution.tsx - Training execution, progress monitoring, status display
- ResultsVisualization.tsx - Results display, charts, metrics visualization, confusion matrix
- StepContent.tsx - Step-based workflow container
- StepIndicator.tsx - Progress indicator for multi-step workflow
- ProgressIndicator.tsx - Generic progress indicator component

### Analytics & Visualization
- AnalyticsTab.tsx - Analytics dashboard with run comparisons, charts, top runs table
- MetadataDisplay.tsx - Displays experiment metadata in organized format

### Chat & Knowledge Base
- ChatSidebar.tsx - AI assistant sidebar with run context, arXiv toggle, session management
- KnowledgeBasePanel.tsx - Knowledge base panel (not read yet)

### Layout Components
- Header.tsx - Application header
- Footer.tsx - Application footer
- WelcomeGuide.tsx - Onboarding guide dialog

### Utility Components
- HelpTooltip.tsx - Help tooltip component (not read yet)
- InstructionCard.tsx - Instruction/info card component (not read yet)
- LoadingOverlay.tsx - Loading overlay component (not read yet)

## Page Import Analysis

### Index.tsx (Main Training Page)
Imports:
- Header, Footer (layout)
- ProgressIndicator, StepContent (workflow)
- DatasetUpload, ExperimentConfig, TrainingExecution, ResultsVisualization (training workflow)
- WelcomeGuide (onboarding)

### SavedExperiments.tsx
Imports:
- Header, Footer (layout)
- AnalyticsTab (analytics)

### Inference.tsx
Imports:
- Header, Footer (layout)
- All components from inference/ subdirectory

### Landing.tsx
Imports:
- Header, Footer (layout)
