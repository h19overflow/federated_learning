# Autonomous Research Assistant - Project Overview

**Project Goal:** Build an AI agent that autonomously explores hyperparameter space, executes experiments via browser automation, and generates research insights.

**Target Impact:** Win "Best Project" award by demonstrating:
- Real autonomous AI research (not just demos)
- Practical time savings (hours vs days of manual tuning)
- Novel integration of LLM reasoning + browser control
- Production-quality engineering with proper modularity

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Research Assistant UI                       │
│         (Live monitoring, experiment history, insights)      │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket
┌────────────────────┴────────────────────────────────────────┐
│               Main Orchestration Layer                       │
│        (Coordinates all modules, manages workflow)           │
└─┬─────────────┬──────────────┬──────────────┬──────────────┘
  │             │              │              │
  │ Module 1    │ Module 2     │ Module 3     │ Module 5
  ▼             ▼              ▼              ▼
┌───────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│ DB &  │  │Research │  │ Browser  │  │ Report   │
│Storage│  │Agent    │  │Automation│  │Generator │
└───────┘  └─────────┘  └──────────┘  └──────────┘
```

---

## Modules & Agent Assignment

| Module | Description | Agent Type | Priority | Depends On |
|--------|-------------|------------|----------|------------|
| **Module 1** | Knowledge Base & Database | backend-logic-architect | P0 | None |
| **Module 2** | Research Orchestrator Agent | backend-logic-architect | P0 | Module 1 (for schema) |
| **Module 3** | Browser Automation Controller | backend-logic-architect | P1 | None |
| **Module 4** | Research Assistant UI | frontend-architect | P1 | Module 1 (for data contract) |
| **Module 5** | Report Generation | backend-logic-architect | P2 | Module 1 |
| **Module 6** | Integration & Main Entry Point | backend-logic-architect | P3 | All modules |

**P0 = Start immediately (parallel), P1 = Start after schemas defined, P2 = Start after core works, P3 = Final integration**

---

## Development Timeline (Estimated)

**Phase 1: Foundation (Days 1-3)**
- Module 1: Database + schemas
- Module 2: Orchestrator agent (can use mock data)
- Module 4: UI scaffold with mock WebSocket

**Phase 2: Core Features (Days 4-7)**
- Module 3: Browser automation
- Module 4: UI completion with real data
- Module 2: Refinement with real experiments

**Phase 3: Polish (Days 8-10)**
- Module 5: Report generation
- Module 6: Integration testing
- End-to-end testing & demo preparation

---

## Module Files

Each module has its own design document:

1. **01-MODULE-1-DATABASE.md** - Knowledge base, schemas, storage
2. **02-MODULE-2-ORCHESTRATOR.md** - LangChain research agent
3. **03-MODULE-3-BROWSER-AUTOMATION.md** - Gemini Computer Use integration
4. **04-MODULE-4-UI.md** - React dashboard components
5. **05-MODULE-5-REPORTS.md** - PDF generation and visualizations
6. **06-MODULE-6-INTEGRATION.md** - Main entry point and orchestration

---

## Interface Contracts (Critical for Parallel Development)

### Module 1 → Module 2
```python
class ExperimentDatabase:
    def get_all_experiments(session_id: int) -> List[ExperimentRun]
    def get_best_result(paradigm: str) -> ExperimentRun
    def save_experiment(session_id: int, proposal: ExperimentProposal, results: dict) -> int
```

### Module 2 → Module 3
```python
class ExperimentProposal(BaseModel):
    proposed_hyperparameters: dict  # {lr, batch_size, dropout, ...}
    reasoning: str
    paradigm: str  # "centralized" | "federated"
    priority: int
```

### Module 3 → Module 1
```python
class ExperimentResults(BaseModel):
    metrics: dict  # {recall, accuracy, f1, auroc, ...}
    training_time_seconds: float
    status: str  # "completed" | "failed"
    error_message: str | None
```

### All Modules → Module 4 (WebSocket Events)
```typescript
type ResearchEvent =
  | { event: "proposal_generated", data: ExperimentProposal }
  | { event: "experiment_started", data: { experiment_id: number } }
  | { event: "experiment_completed", data: ExperimentResults }
  | { event: "agent_reasoning", data: { message: string } }
  | { event: "session_completed", data: { total_experiments: number, best_recall: number } }
```

---

## Key Technologies

**Backend:**
- Google Gemini 2.5 (Computer Use API)
- LangChain + Pydantic (structured output agents)
- Playwright (browser automation)
- SQLAlchemy + SQLite/PostgreSQL
- FastAPI + WebSockets

**Frontend:**
- React + TypeScript
- Your existing UI components (TrainingExecution.tsx patterns)
- WebSocket for real-time updates
- Recharts for visualizations

**Testing:**
- pytest for backend modules
- Vitest for frontend components
- E2E tests with real browser automation

---

## Success Criteria

**Functional:**
- ✅ Agent can autonomously run 30+ experiments without human intervention
- ✅ Browser automation successfully controls your training dashboard
- ✅ Achieves better recall than manual baseline
- ✅ Generates actionable research insights

**Non-Functional:**
- ✅ Modular design allows parallel development by multiple agents
- ✅ Each module has clear interface contracts
- ✅ Comprehensive error handling and recovery
- ✅ Live monitoring dashboard for demo impact

**Demo Impact:**
- ✅ Live demonstration of agent running experiments
- ✅ Visual proof of autonomous reasoning
- ✅ Measurable time/performance improvements
- ✅ Production-quality code and architecture

---

## Next Steps

1. Review all module design documents (01-06)
2. Assign modules to specialized agents
3. Define acceptance criteria for each module
4. Begin parallel development
5. Integration testing once P0/P1 modules complete

---

**Document Version:** 1.0
**Date:** 2026-01-15
**Status:** Ready for Implementation
