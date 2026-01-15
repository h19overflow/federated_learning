# Module 6: Integration & Main Orchestration

**Agent Assignment:** backend-logic-architect
**Priority:** P3 (Final integration after all modules complete)
**Dependencies:** Modules 1-5
**Estimated Effort:** 1-2 days

---

## Purpose

Tie all modules together into a cohesive system with CLI interface, WebSocket server for UI updates, and end-to-end workflow orchestration.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/
├── __init__.py
├── main.py                     # CLI entry point
├── orchestration.py            # Main workflow coordinator
├── websocket_server.py         # WebSocket server for UI
└── config.py                   # Configuration management
```

---

## Main Entry Point (main.py)

```python
import click
import asyncio
from datetime import datetime
from .orchestration import ResearchWorkflow
from .config import ResearchConfig

@click.group()
def cli():
    """Autonomous Research Assistant CLI"""
    pass

@cli.command()
@click.option('--max-experiments', default=30, help='Maximum number of experiments to run')
@click.option('--target-recall', default=0.95, help='Target recall to achieve')
@click.option('--paradigm', type=click.Choice(['both', 'centralized', 'federated']), default='both')
@click.option('--dashboard-url', default='http://localhost:5173', help='Frontend dashboard URL')
@click.option('--headless/--no-headless', default=False, help='Run browser in headless mode')
@click.option('--save-screenshots/--no-screenshots', default=True, help='Save debug screenshots')
@click.option('--max-time-hours', default=24, help='Maximum time budget in hours')
def run(
    max_experiments: int,
    target_recall: float,
    paradigm: str,
    dashboard_url: str,
    headless: bool,
    save_screenshots: bool,
    max_time_hours: int
):
    """
    Run autonomous hyperparameter optimization research

    Example:
        python -m ...research_assistant.main run --max-experiments 30 --target-recall 0.95
    """

    click.echo("🤖 Autonomous Research Assistant")
    click.echo("=" * 50)

    # Build configuration
    config = ResearchConfig(
        max_experiments=max_experiments,
        target_recall=target_recall,
        paradigm=paradigm,
        dashboard_url=dashboard_url,
        browser_headless=headless,
        save_screenshots=save_screenshots,
        max_time_hours=max_time_hours
    )

    click.echo(f"\n📋 Configuration:")
    click.echo(f"  Max Experiments: {max_experiments}")
    click.echo(f"  Target Recall: {target_recall}")
    click.echo(f"  Paradigm: {paradigm}")
    click.echo(f"  Dashboard URL: {dashboard_url}")
    click.echo(f"  Headless Browser: {headless}")
    click.echo(f"\n")

    # Run workflow
    workflow = ResearchWorkflow(config)

    try:
        result = asyncio.run(workflow.run())

        click.echo(f"\n✨ Research Complete!")
        click.echo(f"📊 Total Experiments: {result['total_experiments']}")
        click.echo(f"🏆 Best Centralized Recall: {result['best_centralized_recall']:.3f}")
        click.echo(f"🏆 Best Federated Recall: {result['best_federated_recall']:.3f}")
        click.echo(f"⏱️  Total Time: {result['duration']}")
        click.echo(f"📄 Report: {result['report_path']}")

    except KeyboardInterrupt:
        click.echo("\n⚠️  Research interrupted by user")
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise


@cli.command()
@click.argument('session_id', type=int)
@click.option('--output', default=None, help='Output PDF path')
def report(session_id: int, output: str):
    """
    Generate research report for a completed session

    Example:
        python -m ...research_assistant.main report 123 --output report.pdf
    """

    from .knowledge_base.database import ExperimentDatabase
    from .report_generation.research_report import ResearchReportGenerator

    click.echo(f"📄 Generating report for session {session_id}...")

    db = ExperimentDatabase()
    generator = ResearchReportGenerator(db)

    if not output:
        output = f"research_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    generator.generate_report(session_id, output)

    click.echo(f"✅ Report saved to: {output}")


@cli.command()
@click.option('--port', default=8000, help='WebSocket server port')
def serve(port: int):
    """
    Start WebSocket server for UI communication

    Example:
        python -m ...research_assistant.main serve --port 8000
    """

    from .websocket_server import start_websocket_server

    click.echo(f"🚀 Starting WebSocket server on port {port}...")

    asyncio.run(start_websocket_server(port))


@cli.command()
def list_sessions():
    """List all research sessions"""

    from .knowledge_base.database import ExperimentDatabase

    db = ExperimentDatabase()
    sessions = db.get_all_sessions()

    if not sessions:
        click.echo("No research sessions found.")
        return

    click.echo("\n📊 Research Sessions:")
    click.echo("=" * 80)

    for session in sessions:
        click.echo(f"\nSession ID: {session.id}")
        click.echo(f"  Start: {session.start_time}")
        click.echo(f"  End: {session.end_time or 'In Progress'}")
        click.echo(f"  Experiments: {session.total_experiments}")
        click.echo(f"  Best Centralized: {session.best_centralized_recall or 'N/A'}")
        click.echo(f"  Best Federated: {session.best_federated_recall or 'N/A'}")
        click.echo(f"  Status: {session.stopping_reason or 'Running'}")


if __name__ == '__main__':
    cli()
```

---

## Workflow Orchestration (orchestration.py)

```python
import asyncio
from datetime import datetime
from typing import Dict, Any
import os

from .config import ResearchConfig
from .knowledge_base.database import ExperimentDatabase
from .orchestrator.research_agent import ResearchOrchestrator
from .orchestrator.stopping_criteria import StoppingCriteria
from .browser_automation.controller import BrowserController
from .report_generation.research_report import ResearchReportGenerator
from .websocket_server import broadcast_event

class ResearchWorkflow:
    """
    Main workflow coordinator

    Orchestrates the full research loop:
    1. Initialize all components
    2. For each experiment:
        a. Orchestrator proposes next experiment
        b. Browser controller executes it
        c. Results stored in database
        d. UI updated via WebSocket
    3. Stop when criteria met
    4. Generate final report
    """

    def __init__(self, config: ResearchConfig):
        self.config = config

        # Initialize components
        self.db = ExperimentDatabase()
        self.orchestrator = ResearchOrchestrator()
        self.browser: BrowserController = None
        self.report_generator = ResearchReportGenerator(self.db)

        self.session_id: int = None
        self.start_time: datetime = None

    async def run(self) -> Dict[str, Any]:
        """
        Run the full research workflow

        Returns:
            Summary of research session
        """

        print("🚀 Starting autonomous research session...")
        self.start_time = datetime.now()

        # 1. Initialize session
        self.session_id = self._initialize_session()
        await broadcast_event("session_started", {"session_id": self.session_id})

        # 2. Initialize browser
        self.browser = BrowserController(
            dashboard_url=self.config.dashboard_url,
            headless=self.config.browser_headless,
            save_screenshots=self.config.save_screenshots
        )
        await self.browser.start()

        try:
            # 3. Main research loop
            await self._research_loop()

        finally:
            # 4. Cleanup
            await self.browser.stop()

        # 5. Generate report
        report_path = self._generate_final_report()

        # 6. Return summary
        return self._build_summary(report_path)

    async def _research_loop(self):
        """Main loop: propose → execute → store → check stopping"""

        experiment_count = 0

        while True:
            experiment_count += 1

            print(f"\n{'='*60}")
            print(f"Experiment {experiment_count}/{self.config.max_experiments}")
            print(f"{'='*60}\n")

            # 1. Get experiment history
            history = self.db.get_all_experiments(self.session_id)

            # 2. Check stopping criteria
            should_stop, reason = StoppingCriteria.should_stop(
                history,
                {
                    "max_experiments": self.config.max_experiments,
                    "target_recall": self.config.target_recall,
                    "max_time_hours": self.config.max_time_hours
                }
            )

            if should_stop:
                print(f"\n🛑 Stopping: {reason}")
                self._end_session(reason)
                await broadcast_event("session_completed", {"reason": reason})
                break

            # 3. Orchestrator proposes next experiment
            print("🧠 Agent analyzing history and proposing experiment...")
            proposal = self.orchestrator.analyze_and_propose(
                history,
                self.config.to_dict()
            )

            print(f"   Paradigm: {proposal.paradigm}")
            print(f"   Hyperparameters: {proposal.proposed_hyperparameters}")
            print(f"   Reasoning: {proposal.reasoning}")

            await broadcast_event("proposal_generated", {
                "experiment_number": experiment_count,
                "proposal": proposal.dict()
            })

            # 4. Execute experiment via browser automation
            print(f"\n🌐 Executing experiment via browser...")
            await broadcast_event("experiment_started", {
                "experiment_number": experiment_count,
                "paradigm": proposal.paradigm,
                "hyperparameters": proposal.proposed_hyperparameters
            })

            results = await self.browser.execute_experiment(proposal)

            # 5. Store results
            experiment_id = self.db.save_experiment(
                self.session_id,
                proposal,
                results
            )

            if results.status == "completed":
                print(f"   ✅ Completed: Recall = {results.metrics.get('recall', 0):.3f}")
            else:
                print(f"   ❌ Failed: {results.error_message}")

            await broadcast_event("experiment_completed", {
                "experiment_id": experiment_id,
                "experiment_number": experiment_count,
                "results": results.dict()
            })

            # Update session stats
            self._update_session_stats()

    def _initialize_session(self) -> int:
        """Create new research session in database"""
        return self.db.create_session(self.config.to_dict())

    def _end_session(self, stopping_reason: str):
        """Mark session as complete"""
        self.db.end_session(self.session_id, stopping_reason)

    def _update_session_stats(self):
        """Update session with latest best results"""
        summary = self.db.get_session_summary(self.session_id)
        self.db.update_session_stats(
            self.session_id,
            summary['best_centralized_recall'],
            summary['best_federated_recall']
        )

    def _generate_final_report(self) -> str:
        """Generate and save final PDF report"""
        print("\n📄 Generating final research report...")

        report_path = f"research_report_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.report_generator.generate_report(self.session_id, report_path)

        return report_path

    def _build_summary(self, report_path: str) -> Dict[str, Any]:
        """Build final summary dict"""
        summary = self.db.get_session_summary(self.session_id)

        duration = (datetime.now() - self.start_time).total_seconds()
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)

        return {
            "session_id": self.session_id,
            "total_experiments": summary['total_experiments'],
            "best_centralized_recall": summary['best_centralized_recall'],
            "best_federated_recall": summary['best_federated_recall'],
            "duration": f"{hours}h {minutes}m",
            "report_path": report_path
        }
```

---

## WebSocket Server (websocket_server.py)

```python
import asyncio
import json
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

@app.websocket("/ws/research-progress")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live research updates

    Events sent:
    - session_started
    - proposal_generated
    - experiment_started
    - experiment_completed
    - agent_reasoning
    - session_completed
    """

    await websocket.accept()
    active_connections.add(websocket)

    try:
        # Keep connection alive
        while True:
            # Wait for messages (or just keep alive)
            await websocket.receive_text()

    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_event(event_type: str, data: dict):
    """
    Broadcast event to all connected WebSocket clients

    Args:
        event_type: Type of event (e.g., "experiment_completed")
        data: Event payload
    """

    message = json.dumps({
        "event": event_type,
        "data": data
    })

    # Send to all active connections
    disconnected = set()

    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.add(connection)

    # Remove disconnected clients
    active_connections.difference_update(disconnected)

async def start_websocket_server(port: int = 8000):
    """Start WebSocket server"""
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
```

---

## Configuration (config.py)

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ResearchConfig:
    """Research session configuration"""

    # Experiment limits
    max_experiments: int = 30
    target_recall: float = 0.95
    max_time_hours: int = 24

    # Paradigm selection
    paradigm: str = "both"  # "both", "centralized", "federated"

    # Browser automation
    dashboard_url: str = "http://localhost:5173"
    browser_headless: bool = False
    save_screenshots: bool = True

    # Agent configuration
    agent_model: str = "gemini-2.0-flash-exp"
    agent_temperature: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage"""
        return {
            "max_experiments": self.max_experiments,
            "target_recall": self.target_recall,
            "max_time_hours": self.max_time_hours,
            "paradigm": self.paradigm,
            "dashboard_url": self.dashboard_url,
            "browser_headless": self.browser_headless,
            "save_screenshots": self.save_screenshots,
            "agent_model": self.agent_model,
            "agent_temperature": self.agent_temperature
        }
```

---

## End-to-End Testing

```python
# test_integration.py
import pytest
import asyncio
from .orchestration import ResearchWorkflow
from .config import ResearchConfig

@pytest.mark.asyncio
async def test_full_workflow_small():
    """Test full workflow with minimal experiments"""

    config = ResearchConfig(
        max_experiments=3,
        target_recall=0.80,  # Lower target for testing
        browser_headless=True,
        save_screenshots=False
    )

    workflow = ResearchWorkflow(config)
    result = await workflow.run()

    assert result['total_experiments'] <= 3
    assert result['session_id'] > 0
    assert 'report_path' in result
    assert os.path.exists(result['report_path'])

@pytest.mark.asyncio
async def test_early_stopping_on_target():
    """Test that workflow stops when target achieved"""

    config = ResearchConfig(
        max_experiments=100,
        target_recall=0.85,  # Achievable target
        browser_headless=True
    )

    workflow = ResearchWorkflow(config)
    result = await workflow.run()

    # Should stop before max_experiments if target achieved
    assert result['total_experiments'] < 100
    assert result['best_centralized_recall'] >= 0.85 or result['best_federated_recall'] >= 0.85
```

---

## Usage Examples

**Run full research session:**
```bash
python -m federated_pneumonia_detection.src.control.agentic_systems.research_assistant.main run \
  --max-experiments 30 \
  --target-recall 0.95 \
  --paradigm both
```

**Generate report for existing session:**
```bash
python -m ...research_assistant.main report 123 --output my_report.pdf
```

**Start WebSocket server (separate terminal):**
```bash
python -m ...research_assistant.main serve --port 8000
```

**List all sessions:**
```bash
python -m ...research_assistant.main list-sessions
```

---

## Deployment Checklist

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Ensure training dashboard is running (`npm run dev`)
- [ ] Database initialized (SQLite or PostgreSQL)
- [ ] Python dependencies installed (`uv add google-genai playwright langchain-google-genai`)
- [ ] Playwright browser installed (`playwright install chromium`)
- [ ] WebSocket server running (if using UI monitoring)
- [ ] Firewall allows WebSocket connections (port 8000)

---

## Acceptance Criteria

- ✅ CLI interface works for all commands
- ✅ Full workflow runs end-to-end without errors
- ✅ WebSocket server broadcasts events to UI
- ✅ Browser automation successfully executes experiments
- ✅ Database stores all experiment data
- ✅ Report generation works for completed sessions
- ✅ Stopping criteria correctly trigger
- ✅ Error recovery handles failures gracefully
- ✅ Can resume from interrupted sessions

---

**Status:** Ready for Implementation
**Blocked By:** Modules 1-5
**Blocks:** None (final module)
