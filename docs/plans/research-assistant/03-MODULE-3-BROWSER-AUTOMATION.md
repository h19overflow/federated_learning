# Module 3: Browser Automation Controller

**Agent Assignment:** backend-logic-architect
**Priority:** P1 (Start after Module 1 schemas defined)
**Dependencies:** Google Gemini API key, Playwright, existing training dashboard
**Estimated Effort:** 2-3 days

---

## Purpose

Use Google Gemini 2.5 Computer Use API to autonomously control the training dashboard UI, execute experiments, and extract results.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/browser_automation/
├── __init__.py
├── controller.py           # Main BrowserController class
├── action_executor.py      # Maps Gemini actions to Playwright commands
├── ui_monitor.py           # Extract metrics from UI screenshots
├── error_recovery.py       # Retry logic and error handling
└── screenshots/            # Debug screenshots (gitignored)
```

---

## Core Controller (controller.py)

```python
import asyncio
import base64
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Browser, Page
from google import genai
from google.genai import types
import os
from datetime import datetime

from ..knowledge_base.schemas import ExperimentProposal, ExperimentResults
from .action_executor import ActionExecutor
from .ui_monitor import UIMonitor
from .error_recovery import ErrorRecovery

class BrowserController:
    """
    Controls the training dashboard using Gemini Computer Use API

    Workflow:
    1. Navigate to dashboard
    2. Take screenshot
    3. Ask Gemini to configure experiment
    4. Execute Gemini's action suggestions
    5. Monitor training completion
    6. Extract results
    """

    def __init__(
        self,
        dashboard_url: str = "http://localhost:5173",
        headless: bool = False,
        save_screenshots: bool = True
    ):
        self.dashboard_url = dashboard_url
        self.headless = headless
        self.save_screenshots = save_screenshots

        # Google Gemini client
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.5-computer-use-preview-10-2025"

        # Playwright components (initialized in start())
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Helper classes
        self.action_executor = ActionExecutor()
        self.ui_monitor = UIMonitor()
        self.error_recovery = ErrorRecovery()

        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

    async def start(self):
        """Initialize browser context"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--start-maximized"]
        )

        # Use recommended viewport size for Computer Use
        self.page = await self.browser.new_page(
            viewport={"width": 1440, "height": 900}
        )

        # Navigate to dashboard
        await self.page.goto(f"{self.dashboard_url}/training")
        await self.page.wait_for_load_state("networkidle")

    async def stop(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def execute_experiment(
        self,
        proposal: ExperimentProposal,
        max_retries: int = 3
    ) -> ExperimentResults:
        """
        Execute a single training experiment via browser automation

        Args:
            proposal: Experiment configuration from orchestrator
            max_retries: Number of retry attempts on failure

        Returns:
            ExperimentResults with metrics extracted from UI

        Raises:
            BrowserAutomationError: If experiment fails after retries
        """

        for attempt in range(max_retries):
            try:
                print(f"🚀 Executing experiment (attempt {attempt + 1}/{max_retries})")
                print(f"   Paradigm: {proposal.paradigm}")
                print(f"   Hyperparameters: {proposal.proposed_hyperparameters}")

                # Step 1: Configure experiment
                await self._configure_experiment(proposal)

                # Step 2: Start training
                start_time = datetime.now()
                await self._start_training()

                # Step 3: Monitor until completion
                await self._monitor_training()

                # Step 4: Extract results
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                metrics = await self._extract_results()

                return ExperimentResults(
                    metrics=metrics,
                    training_time_seconds=training_time,
                    status="completed",
                    error_message=None
                )

            except Exception as e:
                print(f"❌ Experiment failed (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:
                    # Try recovery
                    await self.error_recovery.recover(self.page, e)
                else:
                    # Final attempt failed
                    return ExperimentResults(
                        metrics={},
                        training_time_seconds=0.0,
                        status="failed",
                        error_message=str(e)
                    )

    async def _configure_experiment(self, proposal: ExperimentProposal):
        """
        Use Gemini Computer Use to fill in experiment configuration

        Takes screenshot → Gemini suggests actions → Execute actions
        """

        # Take screenshot
        screenshot_bytes = await self.page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

        # Build configuration instruction
        hp = proposal.proposed_hyperparameters
        instruction = f"""Configure the training experiment with these settings:

Training Mode: {proposal.paradigm}
Learning Rate: {hp.get('learning_rate', 0.001)}
Batch Size: {hp.get('batch_size', 32)}
Dropout Rate: {hp.get('dropout_rate', 0.3)}
"""

        if proposal.paradigm == "centralized":
            instruction += f"Epochs: {hp.get('epochs', 10)}\n"
        else:  # federated
            instruction += f"""Local Epochs: {hp.get('local_epochs', 2)}
Federated Rounds: {hp.get('federated_rounds', 5)}
Number of Clients: {hp.get('num_clients', 2)}
"""

        # Ask Gemini to configure
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": instruction},
                    {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}
                ]
            }],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )]
            )
        )

        # Execute Gemini's suggested actions
        if hasattr(response, 'function_calls'):
            for action in response.function_calls:
                await self.action_executor.execute(self.page, action)
                await asyncio.sleep(0.5)  # Small delay between actions

                # Save debug screenshot
                if self.save_screenshots:
                    await self._save_screenshot(f"action_{action.name}")

    async def _start_training(self):
        """
        Click the 'Start Training' button

        Uses Gemini to identify and click the button
        """

        screenshot_bytes = await self.page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": "Click the 'Start Training' button to begin training."},
                    {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}
                ]
            }],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )]
            )
        )

        if hasattr(response, 'function_calls'):
            for action in response.function_calls:
                await self.action_executor.execute(self.page, action)

        # Wait for training to actually start
        await asyncio.sleep(2)

    async def _monitor_training(self, timeout_seconds: int = 600):
        """
        Monitor training until completion or timeout

        Checks training status every 5 seconds via screenshot analysis
        """

        start_time = datetime.now()

        while True:
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Training exceeded {timeout_seconds}s timeout")

            # Take screenshot and check status
            screenshot_bytes = await self.page.screenshot()
            status = await self.ui_monitor.get_training_status(screenshot_bytes)

            print(f"   Status: {status} (elapsed: {elapsed:.0f}s)")

            if status == "completed":
                print("   ✅ Training completed")
                break
            elif status == "error":
                raise RuntimeError("Training failed (error status detected)")
            elif status == "running":
                # Still training, wait
                await asyncio.sleep(5)
            else:
                # Unknown status
                await asyncio.sleep(5)

    async def _extract_results(self) -> Dict[str, float]:
        """
        Extract training metrics from the results UI

        Uses Gemini to read metrics from screenshot
        """

        screenshot_bytes = await self.page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": """Extract the training metrics from this results screen.
Look for: recall, accuracy, precision, F1 score, AUROC.
Return the values as a JSON object like:
{"recall": 0.933, "accuracy": 0.91, "precision": 0.89, "f1": 0.91, "auroc": 0.95}
"""},
                    {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}
                ]
            }]
        )

        # Parse Gemini's response to extract metrics
        metrics = self.ui_monitor.parse_metrics_from_response(response.text)

        if self.save_screenshots:
            await self._save_screenshot("results_extracted")

        return metrics

    async def _save_screenshot(self, name: str):
        """Save screenshot for debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.screenshot_dir}/{timestamp}_{name}.png"
        await self.page.screenshot(path=path)


# Example usage
async def main():
    controller = BrowserController(headless=False)
    await controller.start()

    proposal = ExperimentProposal(
        proposed_hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.3,
            "epochs": 10
        },
        reasoning="Test baseline",
        expected_outcome="Establish baseline performance",
        priority=10,
        paradigm="centralized",
        exploration_phase="broad_exploration"
    )

    results = await controller.execute_experiment(proposal)
    print(f"Results: {results}")

    await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Action Executor (action_executor.py)

```python
from playwright.async_api import Page
from typing import Any

class ActionExecutor:
    """
    Maps Gemini Computer Use actions to Playwright commands

    Gemini uses normalized 1000x1000 coordinates
    We need to scale to actual viewport (1440x900)
    """

    def __init__(self):
        self.viewport_width = 1440
        self.viewport_height = 900

    async def execute(self, page: Page, action: Any):
        """
        Execute a Gemini computer use action

        Supported actions:
        - click_at(x, y)
        - type_text_at(x, y, text)
        - scroll_document(direction, amount)
        - navigate(url)
        - wait_5_seconds()
        """

        action_name = action.name
        args = action.args

        if action_name == "click_at":
            await self._click_at(page, args["x"], args["y"])

        elif action_name == "type_text_at":
            await self._type_text_at(page, args["x"], args["y"], args["text"])

        elif action_name == "scroll_document":
            await self._scroll_document(page, args["direction"], args.get("amount", 100))

        elif action_name == "navigate":
            await page.goto(args["url"])

        elif action_name == "wait_5_seconds":
            await asyncio.sleep(5)

        elif action_name == "key_combination":
            # Handle keyboard shortcuts
            await page.keyboard.press(args["keys"])

        else:
            print(f"⚠️  Unknown action: {action_name}")

    async def _click_at(self, page: Page, x_norm: int, y_norm: int):
        """
        Click at normalized coordinates (1000x1000 → 1440x900)
        """
        x = (x_norm / 1000) * self.viewport_width
        y = (y_norm / 1000) * self.viewport_height

        await page.mouse.click(x, y)

    async def _type_text_at(self, page: Page, x_norm: int, y_norm: int, text: str):
        """
        Click at position, then type text
        """
        await self._click_at(page, x_norm, y_norm)
        await asyncio.sleep(0.2)  # Wait for focus
        await page.keyboard.type(text)

    async def _scroll_document(self, page: Page, direction: str, amount: int):
        """
        Scroll the page
        """
        if direction == "down":
            await page.mouse.wheel(0, amount)
        elif direction == "up":
            await page.mouse.wheel(0, -amount)
```

---

## UI Monitor (ui_monitor.py)

```python
import re
import json
from typing import Dict, Any

class UIMonitor:
    """
    Analyze UI screenshots to extract training status and metrics

    Uses heuristics + Gemini vision to understand the UI state
    """

    async def get_training_status(self, screenshot_bytes: bytes) -> str:
        """
        Determine training status from screenshot

        Returns: "idle" | "running" | "completed" | "error"
        """

        # Use Gemini to analyze screenshot
        # (Simplified - in practice, use Gemini vision API)

        # Look for UI indicators:
        # - "Training" badge → running
        # - "Completed" badge → completed
        # - "Error" badge → error
        # - "Ready" badge → idle

        # For now, placeholder
        return "running"

    def parse_metrics_from_response(self, gemini_response: str) -> Dict[str, float]:
        """
        Parse metrics from Gemini's text response

        Gemini should return JSON, but may include extra text
        """

        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', gemini_response)
        if json_match:
            try:
                metrics = json.loads(json_match.group())
                return metrics
            except json.JSONDecodeError:
                pass

        # Fallback: parse individual metrics
        metrics = {}

        patterns = {
            "recall": r"recall[\":\s]+([\d.]+)",
            "accuracy": r"accuracy[\":\s]+([\d.]+)",
            "precision": r"precision[\":\s]+([\d.]+)",
            "f1": r"f1[\":\s]+([\d.]+)",
            "auroc": r"auroc[\":\s]+([\d.]+)"
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, gemini_response, re.IGNORECASE)
            if match:
                metrics[metric] = float(match.group(1))

        return metrics
```

---

## Error Recovery (error_recovery.py)

```python
from playwright.async_api import Page
import asyncio

class ErrorRecovery:
    """
    Handle common errors and retry strategies
    """

    async def recover(self, page: Page, error: Exception):
        """
        Attempt to recover from error
        """

        error_type = type(error).__name__

        if "timeout" in error_type.lower():
            # Reload page
            print("   🔄 Timeout detected, reloading page...")
            await page.reload()
            await page.wait_for_load_state("networkidle")

        elif "element not found" in str(error).lower():
            # UI changed, take new screenshot
            print("   🔄 UI element not found, retrying...")
            await asyncio.sleep(2)

        else:
            # Generic retry
            print(f"   🔄 Error: {error}, waiting before retry...")
            await asyncio.sleep(5)
```

---

## Testing Strategy

```python
# test_browser_automation.py
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_configure_experiment():
    """Test that configuration actions are executed"""
    controller = BrowserController(headless=True)
    await controller.start()

    proposal = ExperimentProposal(...)

    # Mock Gemini response
    controller.gemini_client.models.generate_content = AsyncMock(
        return_value=Mock(function_calls=[...])
    )

    await controller._configure_experiment(proposal)

    # Verify actions were executed
    # ...

    await controller.stop()

@pytest.mark.asyncio
async def test_extract_results():
    """Test metric extraction from UI"""
    monitor = UIMonitor()

    gemini_response = '{"recall": 0.933, "accuracy": 0.91}'
    metrics = monitor.parse_metrics_from_response(gemini_response)

    assert metrics["recall"] == 0.933
    assert metrics["accuracy"] == 0.91
```

---

## Acceptance Criteria

- ✅ Browser automation successfully navigates to dashboard
- ✅ Gemini Computer Use correctly fills in configuration
- ✅ Training can be started via button click
- ✅ Status monitoring detects completion
- ✅ Metrics are accurately extracted from results UI
- ✅ Error recovery handles common failures
- ✅ Screenshots saved for debugging
- ✅ Works with both centralized and federated modes

---

**Status:** Ready for Implementation
**Blocked By:** None (can test against existing dashboard)
**Blocks:** Module 6 (integration)
