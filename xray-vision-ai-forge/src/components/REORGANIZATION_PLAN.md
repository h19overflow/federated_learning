# Component Reorganization Plan

## Current State Analysis

### Problem

Components are scattered at the root level of `components/` without logical grouping, making it hard to find related components and understand the codebase structure.

### Current Structure

```
components/
├── AnalyticsTab.tsx          (root - loose)
├── ChatSidebar.tsx           (root - loose)
├── DatasetUpload.tsx         (root - DUPLICATE)
├── ExperimentConfig.tsx      (root - DUPLICATE)
├── Footer.tsx                (root - loose)
├── Header.tsx                (root - loose)
├── HelpTooltip.tsx           (root - loose)
├── InstructionCard.tsx       (root - loose)
├── KnowledgeBasePanel.tsx    (root - loose)
├── LoadingOverlay.tsx        (root - loose)
├── MetadataDisplay.tsx       (root - loose)
├── ProgressIndicator.tsx     (root - DUPLICATE)
├── ResultsVisualization.tsx  (root - DUPLICATE)
├── StepContent.tsx           (root - DUPLICATE)
├── StepIndicator.tsx         (root - DUPLICATE)
├── TrainingExecution.tsx     (root - DUPLICATE)
├── WelcomeGuide.tsx          (root - loose)
├── inference/                (organized - 11 files)
├── observability/            (organized - 3 files)
├── training/                 (organized - 7 files, but UNUSED duplicates)
└── ui/                       (shadcn primitives - 49 files)
```

### Key Findings

1. **17 loose components** at root level
2. **7 duplicate files** exist in both root AND `training/` folder
3. **Pages import from ROOT**, not from `training/` folder
4. The `training/` folder files appear to be UNUSED duplicates
5. `inference/` and `observability/` are already well-organized with index.ts

---

## Proposed Structure

```
components/
├── layout/                   # App shell & navigation (NEW)
│   ├── Header.tsx
│   ├── Footer.tsx
│   └── index.ts
│
├── training/                 # Training workflow (CONSOLIDATE)
│   ├── DatasetUpload.tsx
│   ├── ExperimentConfig.tsx
│   ├── TrainingExecution.tsx
│   ├── ResultsVisualization.tsx
│   ├── StepContent.tsx
│   ├── StepIndicator.tsx
│   ├── ProgressIndicator.tsx
│   └── index.ts
│
├── analytics/                # Analytics & data display (NEW)
│   ├── AnalyticsTab.tsx
│   ├── MetadataDisplay.tsx
│   └── index.ts
│
├── chat/                     # Chat & knowledge base (NEW)
│   ├── ChatSidebar.tsx
│   ├── KnowledgeBasePanel.tsx
│   └── index.ts
│
├── shared/                   # Shared/utility components (NEW)
│   ├── HelpTooltip.tsx
│   ├── InstructionCard.tsx
│   ├── LoadingOverlay.tsx
│   ├── WelcomeGuide.tsx
│   └── index.ts
│
├── inference/                # KEEP AS-IS (already organized)
│   └── (11 files + index.ts)
│
├── observability/            # KEEP AS-IS (already organized)
│   └── (3 files + index.ts)
│
└── ui/                       # KEEP AS-IS (shadcn primitives)
    └── (49 files)
```

---

## File Movement Plan

### Phase 1: Create New Directories

- Create `layout/`
- Create `analytics/`
- Create `chat/`
- Create `shared/`

### Phase 2: Move Components

| Current Location           | New Location                        | Action                            |
| -------------------------- | ----------------------------------- | --------------------------------- |
| `Header.tsx`               | `layout/Header.tsx`                 | MOVE                              |
| `Footer.tsx`               | `layout/Footer.tsx`                 | MOVE                              |
| `DatasetUpload.tsx`        | `training/DatasetUpload.tsx`        | MOVE (replace existing duplicate) |
| `ExperimentConfig.tsx`     | `training/ExperimentConfig.tsx`     | MOVE (replace existing duplicate) |
| `TrainingExecution.tsx`    | `training/TrainingExecution.tsx`    | MOVE (replace existing duplicate) |
| `ResultsVisualization.tsx` | `training/ResultsVisualization.tsx` | MOVE (replace existing duplicate) |
| `StepContent.tsx`          | `training/StepContent.tsx`          | MOVE (replace existing duplicate) |
| `StepIndicator.tsx`        | `training/StepIndicator.tsx`        | MOVE (replace existing duplicate) |
| `ProgressIndicator.tsx`    | `training/ProgressIndicator.tsx`    | MOVE (replace existing duplicate) |
| `AnalyticsTab.tsx`         | `analytics/AnalyticsTab.tsx`        | MOVE                              |
| `MetadataDisplay.tsx`      | `analytics/MetadataDisplay.tsx`     | MOVE                              |
| `ChatSidebar.tsx`          | `chat/ChatSidebar.tsx`              | MOVE                              |
| `KnowledgeBasePanel.tsx`   | `chat/KnowledgeBasePanel.tsx`       | MOVE                              |
| `HelpTooltip.tsx`          | `shared/HelpTooltip.tsx`            | MOVE                              |
| `InstructionCard.tsx`      | `shared/InstructionCard.tsx`        | MOVE                              |
| `LoadingOverlay.tsx`       | `shared/LoadingOverlay.tsx`         | MOVE                              |
| `WelcomeGuide.tsx`         | `shared/WelcomeGuide.tsx`           | MOVE                              |

### Phase 3: Create Index Files

Add `index.ts` barrel exports to each new folder for clean imports.

### Phase 4: Update Page Imports

Update all import paths in pages:

| Page                   | Changes Required                                                     |
| ---------------------- | -------------------------------------------------------------------- |
| `Index.tsx`            | Update 9 imports (Header, Footer, training components, WelcomeGuide) |
| `Inference.tsx`        | Update 2 imports (Header, Footer)                                    |
| `Landing.tsx`          | Update 2 imports (Header, Footer)                                    |
| `SavedExperiments.tsx` | Update 3 imports (Header, Footer, AnalyticsTab)                      |

### Phase 5: Update Cross-Component Imports

Check and update any internal component imports.

---

## Import Path Changes Summary

### Before

```typescript
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import DatasetUpload from "@/components/DatasetUpload";
import AnalyticsTab from "@/components/AnalyticsTab";
```

### After

```typescript
import { Header, Footer } from "@/components/layout";
import { DatasetUpload } from "@/components/training";
import { AnalyticsTab } from "@/components/analytics";
```

Or with direct imports:

```typescript
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import DatasetUpload from "@/components/training/DatasetUpload";
import AnalyticsTab from "@/components/analytics/AnalyticsTab";
```

---

## Cleanup

- Delete duplicate files in `training/` folder (after verifying root files are the correct versions)
- Ensure no orphaned imports remain

---

## Benefits

1. **Clear functional grouping** - Related components are co-located
2. **Improved discoverability** - Easy to find components by purpose
3. **Better maintainability** - Changes to one area don't affect unrelated areas
4. **Consistent with existing patterns** - Matches `inference/` and `observability/` organization
5. **Clean barrel exports** - `index.ts` files enable shorter imports

---

## Risk Assessment

- **LOW RISK**: This is purely file movement and import updates
- **NO CODE CHANGES**: Component logic remains unchanged
- **REVERSIBLE**: Git tracks all changes

---

## Verification Steps

1. Run TypeScript compilation to catch import errors
2. Run the development server to verify app loads
3. Navigate through all pages to ensure functionality
