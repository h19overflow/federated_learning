# Autonomous Research Assistant - Design Documents

**Date:** 2026-01-15
**Status:** Ready for Implementation
**Target:** Win "Best Project" Award

---

## Quick Start

1. **Read Overview:** Start with `00-OVERVIEW.md` for architecture and module breakdown
2. **Assign Modules:** Distribute modules 1-6 to specialized agents (see table below)
3. **Parallel Development:** Modules 1-3 can start immediately
4. **Integration:** Module 6 ties everything together after core modules complete

---

## Module Overview

| Module | File | Agent | Priority | Can Start |
|--------|------|-------|----------|-----------|
| **Module 1** | 01-MODULE-1-DATABASE.md | backend-logic-architect | P0 | ✅ Now |
| **Module 2** | 02-MODULE-2-ORCHESTRATOR.md | backend-logic-architect | P0 | ✅ Now |
| **Module 3** | 03-MODULE-3-BROWSER-AUTOMATION.md | backend-logic-architect | P1 | ✅ Now |
| **Module 4** | 04-MODULE-4-UI.md | frontend-architect | P1 | After Module 1 schemas |
| **Module 5** | 05-MODULE-5-REPORTS.md | backend-logic-architect | P2 | After Module 1 complete |
| **Module 6** | 06-MODULE-6-INTEGRATION.md | backend-logic-architect | P3 | After all modules |

---

## Key Design Decisions

### Why Google Gemini Computer Use?
- Native browser automation (no brittle selectors)
- 1000×1000 normalized coordinates (screen-size agnostic)
- Built-in screenshot analysis
- 14 predefined UI actions
- Better than Anthropic's for this use case (web-focused)

### Why LangChain Agents?
- Follows your existing project patterns (`query_router.py`, `title_generator.py`)
- Structured output with Pydantic (type-safe)
- Easy to test and iterate
- Gemini 2.0 Flash integration

### Why Modular Architecture?
- Enables parallel development by multiple agents
- Clear interface contracts between modules
- Each module can be tested independently
- Easy to replace/upgrade components

### Why SQLite/PostgreSQL?
- Persistent storage for experiment history
- Easy analytics queries
- Supports JSON fields for flexibility
- Can migrate from SQLite (dev) to PostgreSQL (production)

---

## Dependencies

**Python Packages:**
```bash
uv add google-genai playwright langchain-google-genai sqlalchemy fastapi uvicorn websockets reportlab matplotlib seaborn
playwright install chromium
```

**Environment Variables:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Frontend Dependencies:**
Already installed in `xray-vision-ai-forge` (React, TypeScript, WebSocket)

---

## Implementation Order

### Phase 1: Foundation (Days 1-3)
1. **Module 1** (Database): Define schemas, CRUD operations, FastAPI endpoints
2. **Module 2** (Orchestrator): Build LangChain agent with structured output
3. **Module 3** (Browser Automation): Get basic Gemini Computer Use working

**Deliverable:** Can propose experiments and execute them manually

### Phase 2: Core Features (Days 4-7)
4. **Module 4** (UI): Build React dashboard with WebSocket updates
5. Refine Module 2 with smarter exploration strategies
6. Improve Module 3 error recovery and monitoring

**Deliverable:** End-to-end workflow (agent proposes → browser executes → UI updates)

### Phase 3: Polish (Days 8-10)
7. **Module 5** (Reports): PDF generation with visualizations
8. **Module 6** (Integration): CLI, WebSocket server, full workflow
9. End-to-end testing and demo preparation

**Deliverable:** Production-ready system with impressive demo

---

## Testing Strategy

**Unit Tests:**
- Module 1: Database CRUD operations
- Module 2: Agent proposal logic, stopping criteria
- Module 3: Action executor, UI monitor
- Module 4: React component rendering
- Module 5: Report generation, visualizations

**Integration Tests:**
- Module 6: Full workflow with minimal experiments (3-5 runs)
- WebSocket event broadcasting
- Database → UI data flow

**E2E Tests:**
- Full 30-experiment research session
- Both centralized and federated paradigms
- Report generation for completed sessions

---

## Demo Strategy

### Setup (Before Demo)
1. Run overnight research session (30 experiments)
2. Generate beautiful PDF report
3. Prepare talking points about insights discovered

### Live Demo Flow
1. **Problem Introduction** (1 min)
   - "Hyperparameter tuning is tedious and requires expertise"
   - Show manual baseline: 0.921 recall after 10 hours of work

2. **Agent Introduction** (1 min)
   - "Our autonomous agent uses Gemini Computer Use + LangChain"
   - Show architecture diagram (3 components)

3. **Live Execution** (3 min)
   - Start new research session with 5 experiments
   - Show agent reasoning in real-time
   - Show browser automation in action
   - Show UI updating with results

4. **Results Reveal** (2 min)
   - Show overnight session results: 25 experiments in 6 hours
   - Best recall improved to 0.947 (2.6% improvement)
   - Show PDF report with insights
   - Highlight agent discovered: "Federated needs 2× higher LR"

5. **Impact Summary** (1 min)
   - Time saved: 6 hours vs 3 days manual
   - Performance gain: 2.6% improvement
   - Zero human intervention required
   - Production-quality engineering

### Wow Factor Elements
✨ **Live browser automation** (judges can see it working)
✨ **Agent reasoning displayed** (shows intelligence, not randomness)
✨ **Real improvements** (not just a demo, actual research value)
✨ **Beautiful visualizations** (professional quality)
✨ **Modular architecture** (demonstrates engineering skill)

---

## Success Metrics

**Functional:**
- ✅ Agent runs 30+ experiments autonomously
- ✅ Browser automation success rate > 90%
- ✅ Finds hyperparameters better than manual baseline
- ✅ Generates actionable research insights

**Non-Functional:**
- ✅ Modular design enables parallel development
- ✅ Comprehensive test coverage (>80%)
- ✅ Professional documentation
- ✅ Production-ready code quality

**Demo Impact:**
- ✅ Judges say "wow" during live demo
- ✅ Clear competitive advantage vs other projects
- ✅ Demonstrates practical real-world value
- ✅ Shows both AI/ML and software engineering skills

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Browser automation fails | Extensive error recovery, retry logic, screenshots for debugging |
| Agent proposes bad experiments | Validation layer, stop on errors, human-in-loop option |
| Demo breaks during presentation | Pre-record backup video, have completed session to show |
| Time overrun | Prioritize P0/P1 modules, P2/P3 are enhancements |
| API rate limits | Cache responses, implement exponential backoff |

---

## Next Steps

1. ✅ Review all design documents with team
2. ✅ Set up GEMINI_API_KEY
3. ✅ Create feature branch: `feature/autonomous-research-assistant`
4. ✅ Assign modules to agents using Task tool
5. ✅ Begin parallel development (Modules 1-3)
6. ✅ Daily standup to track progress
7. ✅ Integration testing after Phase 1 complete

---

## Questions?

Refer to individual module files for detailed specifications:
- **Architecture:** 00-OVERVIEW.md
- **Database:** 01-MODULE-1-DATABASE.md
- **Agent Logic:** 02-MODULE-2-ORCHESTRATOR.md
- **Browser Control:** 03-MODULE-3-BROWSER-AUTOMATION.md
- **UI Dashboard:** 04-MODULE-4-UI.md
- **Reports:** 05-MODULE-5-REPORTS.md
- **Integration:** 06-MODULE-6-INTEGRATION.md

---

**Let's build something amazing! 🚀**
