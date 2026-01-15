# Module 2: Research Orchestrator Agent

**Agent Assignment:** backend-logic-architect
**Priority:** P0 (Start immediately, can use mock data)
**Dependencies:** Module 1 schemas (for data contracts)
**Estimated Effort:** 2-3 days

---

## Purpose

The "brain" of the research assistant. Analyzes experiment history and proposes the next experiment using LangChain agents with structured output.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/orchestrator/
├── __init__.py
├── research_agent.py           # Main LangChain agent
├── experiment_planner.py       # Bayesian-inspired strategy
├── stopping_criteria.py        # Convergence detection
└── prompts/
    ├── system_prompt.txt       # Agent system instructions
    └── analysis_prompt.txt     # Few-shot examples
```

---

## Research Agent (research_agent.py)

**Pattern:** Follow your existing LangChain patterns from `query_router.py` and `title_generator.py`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

class ExperimentProposal(BaseModel):
    """Structured output from agent"""
    proposed_hyperparameters: Dict[str, Any] = Field(
        description="Next hyperparameters to test",
        example={"learning_rate": 0.001, "batch_size": 32, "dropout_rate": 0.3}
    )
    reasoning: str = Field(
        description="Detailed reasoning for this choice",
        example="Based on trend analysis, recall improved by 2% when lr decreased from 0.01 to 0.001. Testing lr=0.0005 to see if this trend continues."
    )
    expected_outcome: str = Field(
        description="What we expect to learn from this experiment"
    )
    priority: int = Field(
        ge=1, le=10,
        description="Priority 1-10, higher means more important to test next"
    )
    paradigm: str = Field(
        description="Which training paradigm to test",
        pattern="^(centralized|federated)$"
    )
    exploration_phase: str = Field(
        description="Which exploration phase this belongs to",
        pattern="^(broad_exploration|smart_refinement|fine_tuning)$"
    )

class ResearchOrchestrator:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,  # Balance creativity and consistency
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.agent = create_agent(
            self.model,
            tools=[],  # No external tools needed, pure reasoning
            system_prompt=self._load_system_prompt()
        )

        self.current_phase = "broad_exploration"

    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        # See prompts/system_prompt.txt below
        pass

    def analyze_and_propose(
        self,
        experiment_history: List[Dict[str, Any]],
        session_config: Dict[str, Any]
    ) -> ExperimentProposal:
        """
        Main entry point: analyze history and propose next experiment

        Args:
            experiment_history: List of previous experiments with results
            session_config: Session settings (max_experiments, target_recall, etc.)

        Returns:
            ExperimentProposal with next experiment to run
        """

        # Build context from history
        context = self._build_analysis_context(experiment_history)

        # Determine current exploration phase
        self.current_phase = self._determine_phase(experiment_history, session_config)

        # Invoke agent with structured output
        result = self.agent.invoke({
            "messages": [{
                "role": "user",
                "content": self._build_prompt(context, session_config)
            }]
        }, config={"response_format": ExperimentProposal})

        return result.get("structured_response")

    def _build_analysis_context(self, history: List[Dict]) -> str:
        """Build formatted context from experiment history"""

        if not history:
            return "No experiments run yet. This is the first experiment."

        # Format history for agent
        context_lines = [
            f"Total experiments run: {len(history)}",
            f"\nRecent experiments (last 5):"
        ]

        for exp in history[-5:]:
            paradigm = exp["paradigm"]
            hp = exp["hyperparameters"]
            metrics = exp.get("metrics", {})
            recall = metrics.get("recall", "N/A")
            status = exp["status"]

            context_lines.append(
                f"  - Exp #{exp['experiment_number']} [{paradigm}]: "
                f"lr={hp.get('learning_rate')}, batch={hp.get('batch_size')}, "
                f"dropout={hp.get('dropout_rate')} → Recall={recall} ({status})"
            )

        # Add best results
        best_cent = self._get_best_by_paradigm(history, "centralized")
        best_fed = self._get_best_by_paradigm(history, "federated")

        context_lines.append(f"\nBest centralized recall: {best_cent}")
        context_lines.append(f"Best federated recall: {best_fed}")

        # Add trend analysis
        trends = self._analyze_trends(history)
        context_lines.append(f"\nTrend Analysis:\n{trends}")

        return "\n".join(context_lines)

    def _build_prompt(self, context: str, config: Dict) -> str:
        """Build the actual prompt for the agent"""

        prompt = f"""You are an autonomous AI research assistant optimizing hyperparameters for pneumonia detection.

PRIMARY GOAL: Maximize recall (minimize false negatives) while maintaining accuracy > 0.85
SECONDARY GOAL: Compare centralized vs federated learning performance

Current Exploration Phase: {self.current_phase}
Max Experiments Allowed: {config.get('max_experiments', 30)}
Target Recall: {config.get('target_recall', 0.95)}

{context}

HYPERPARAMETER SPACE:
- learning_rate: [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
- batch_size: [8, 16, 32, 64, 128, 256]
- dropout_rate: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- epochs (centralized): [5, 10, 15, 20]
- local_epochs (federated): [1, 2, 3, 5]
- num_clients (federated): [2, 3, 4, 5]

STRATEGY BY PHASE:
- broad_exploration: Test corners of hyperparameter space, explore widely
- smart_refinement: Focus on promising regions identified in exploration
- fine_tuning: Combine best hyperparameters, test with multiple seeds

Based on the experiment history, propose the NEXT BEST EXPERIMENT to run.
Consider:
1. Which hyperparameters have the biggest impact on recall?
2. Are there unexplored regions of the parameter space?
3. Do centralized and federated need different hyperparameters?
4. What trends do you observe (monotonic improvements, interactions)?
5. Are we converging or should we explore more?

Provide a detailed proposal with reasoning.
"""
        return prompt

    def _determine_phase(self, history: List[Dict], config: Dict) -> str:
        """Determine which exploration phase we're in"""

        total_experiments = len(history)
        max_experiments = config.get("max_experiments", 30)

        if total_experiments < 8:
            return "broad_exploration"
        elif total_experiments < max_experiments * 0.7:
            return "smart_refinement"
        else:
            return "fine_tuning"

    def _get_best_by_paradigm(self, history: List[Dict], paradigm: str) -> float:
        """Get best recall for a paradigm"""
        paradigm_exps = [e for e in history if e["paradigm"] == paradigm and e["status"] == "completed"]
        if not paradigm_exps:
            return 0.0
        return max(e["metrics"]["recall"] for e in paradigm_exps if "metrics" in e)

    def _analyze_trends(self, history: List[Dict]) -> str:
        """Analyze trends in experiment results"""
        # TODO: Implement trend analysis
        # - Correlation between hyperparameters and recall
        # - Monotonic trends (e.g., recall always improves as lr decreases)
        # - Interaction effects (e.g., dropout works better with lower lr)

        return "Trend analysis placeholder"
```

---

## Experiment Planner (experiment_planner.py)

**Purpose:** Helper functions for intelligent experiment selection

```python
from typing import List, Dict, Any, Tuple
import numpy as np

class ExperimentPlanner:
    """Bayesian-inspired hyperparameter exploration strategies"""

    @staticmethod
    def suggest_broad_exploration_points(
        hyperparameter_space: Dict[str, List],
        already_tested: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Suggest corner points and important regions for initial exploration

        Uses Latin Hypercube Sampling for good coverage
        """
        pass

    @staticmethod
    def identify_promising_regions(
        history: List[Dict],
        metric: str = "recall"
    ) -> Dict[str, Tuple[float, float]]:
        """
        Identify which hyperparameter ranges show promise

        Returns: {"learning_rate": (0.0005, 0.002), ...}
        """
        pass

    @staticmethod
    def suggest_refinement_points(
        promising_regions: Dict[str, Tuple],
        already_tested: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Suggest points within promising regions for refinement
        """
        pass

    @staticmethod
    def detect_hyperparameter_interactions(
        history: List[Dict],
        param1: str,
        param2: str
    ) -> Dict[str, Any]:
        """
        Detect if two hyperparameters interact

        Example: "dropout=0.5 works well with lr=0.01 but not lr=0.001"
        """
        pass

    @staticmethod
    def calculate_expected_improvement(
        proposed_config: Dict[str, Any],
        history: List[Dict],
        best_recall: float
    ) -> float:
        """
        Estimate likelihood that this config will improve on best_recall

        Uses simple heuristic based on similarity to best configs
        """
        pass
```

---

## Stopping Criteria (stopping_criteria.py)

```python
from typing import List, Dict
import numpy as np

class StoppingCriteria:
    """Determine when to stop the research session"""

    @staticmethod
    def has_converged(
        recent_experiments: List[Dict],
        window_size: int = 5,
        threshold: float = 0.005  # 0.5% improvement
    ) -> bool:
        """
        Detect convergence: last N experiments show < threshold improvement

        Args:
            recent_experiments: Last N experiments (sorted by time)
            window_size: How many recent experiments to check
            threshold: Minimum improvement to consider significant

        Returns:
            True if converged (stop searching)
        """

        if len(recent_experiments) < window_size:
            return False

        completed = [e for e in recent_experiments[-window_size:] if e["status"] == "completed"]
        if len(completed) < window_size:
            return False

        recalls = [e["metrics"]["recall"] for e in completed]

        # Check if max improvement in window is < threshold
        max_recall = max(recalls)
        min_recall = min(recalls)
        improvement = max_recall - min_recall

        return improvement < threshold

    @staticmethod
    def target_achieved(
        best_recall: float,
        target_recall: float,
        margin: float = 0.01
    ) -> bool:
        """
        Check if target recall has been achieved

        Args:
            best_recall: Best recall achieved so far
            target_recall: Target to achieve
            margin: Allow slight margin (target - margin is acceptable)
        """
        return best_recall >= (target_recall - margin)

    @staticmethod
    def should_stop(
        experiment_history: List[Dict],
        session_config: Dict
    ) -> Tuple[bool, str]:
        """
        Main stopping criteria check

        Returns:
            (should_stop: bool, reason: str)
        """

        max_experiments = session_config.get("max_experiments", 30)
        target_recall = session_config.get("target_recall", 0.95)
        max_time_hours = session_config.get("max_time_hours", 24)

        # Check max experiments
        if len(experiment_history) >= max_experiments:
            return True, f"Maximum experiments ({max_experiments}) reached"

        # Check target achieved
        completed = [e for e in experiment_history if e["status"] == "completed"]
        if completed:
            best_recall = max(e["metrics"]["recall"] for e in completed)
            if StoppingCriteria.target_achieved(best_recall, target_recall):
                return True, f"Target recall {target_recall:.3f} achieved ({best_recall:.3f})"

        # Check convergence
        if StoppingCriteria.has_converged(experiment_history):
            return True, "Convergence detected (< 0.5% improvement in last 5 experiments)"

        # Check time budget
        if experiment_history:
            elapsed_hours = (experiment_history[-1]["timestamp"] - experiment_history[0]["timestamp"]).total_seconds() / 3600
            if elapsed_hours >= max_time_hours:
                return True, f"Time budget ({max_time_hours}h) exceeded"

        return False, ""
```

---

## System Prompt (prompts/system_prompt.txt)

```
You are an expert AI research assistant specializing in hyperparameter optimization for medical AI models.

Your task is to autonomously explore the hyperparameter space to maximize recall (minimize false negatives) for pneumonia detection from chest X-rays.

CRITICAL DOMAIN KNOWLEDGE:
- False negatives in medical AI are more costly than false positives
- Recall > 0.95 is the gold standard for clinical deployment
- Accuracy must remain above 0.85 to be useful
- Federated learning may require different hyperparameters than centralized
- Class imbalance: 68% normal, 32% pneumonia

HYPERPARAMETER INTUITIONS:
- Learning rate: Too high → unstable, too low → slow convergence
- Batch size: Smaller → better generalization but slower training
- Dropout: Higher → more regularization, prevents overfitting
- Epochs: More is better until overfitting starts

EXPLORATION STRATEGY:
1. Broad Exploration (first ~8 experiments):
   - Test corners of hyperparameter space
   - Identify which parameters matter most
   - Establish baseline performance

2. Smart Refinement (middle ~15 experiments):
   - Focus on promising regions
   - Test parameter interactions
   - Compare centralized vs federated carefully

3. Fine-Tuning (final ~7 experiments):
   - Combine best hyperparameters
   - Test with multiple random seeds for robustness
   - Validate best configurations

REASONING STYLE:
- Be data-driven: base decisions on observed trends
- Be explicit: clearly state assumptions and reasoning
- Be strategic: don't waste experiments on unlikely configs
- Be scientific: form hypotheses and test them

AVOID:
- Random search (be intelligent)
- Redundant experiments (check history first)
- Ignoring paradigm differences (centralized ≠ federated)
- Stopping too early (explore thoroughly)
```

---

## Testing Strategy

```python
# test_orchestrator.py
def test_first_experiment_proposal():
    """Agent should propose broad exploration when no history"""
    orchestrator = ResearchOrchestrator()
    proposal = orchestrator.analyze_and_propose([], {"max_experiments": 30})

    assert proposal.paradigm in ["centralized", "federated"]
    assert proposal.exploration_phase == "broad_exploration"
    assert len(proposal.reasoning) > 50  # Should provide detailed reasoning

def test_refinement_phase():
    """Agent should refine after initial exploration"""
    # Mock history with 8 experiments showing lr trend
    history = [...]  # Mock data
    orchestrator = ResearchOrchestrator()
    proposal = orchestrator.analyze_and_propose(history, {"max_experiments": 30})

    assert proposal.exploration_phase == "smart_refinement"

def test_convergence_detection():
    """Should detect when experiments plateau"""
    history = [
        # 5 experiments with similar recall (~0.93)
    ]
    should_stop, reason = StoppingCriteria.should_stop(history, {"max_experiments": 30})

    assert should_stop
    assert "convergence" in reason.lower()

def test_target_achieved():
    """Should stop when target reached"""
    history = [{"metrics": {"recall": 0.96}, "status": "completed"}]
    should_stop, reason = StoppingCriteria.should_stop(history, {"target_recall": 0.95})

    assert should_stop
    assert "target" in reason.lower()
```

---

## Acceptance Criteria

- ✅ Agent generates valid ExperimentProposal with structured output
- ✅ Proposals are data-driven (reasoning references experiment history)
- ✅ Phase transitions work correctly (exploration → refinement → fine-tuning)
- ✅ Stopping criteria detect convergence, target achievement, limits
- ✅ Agent avoids redundant experiments (checks history)
- ✅ Works with both centralized and federated paradigms
- ✅ Unit tests cover key scenarios (first experiment, refinement, convergence)

---

**Status:** Ready for Implementation
**Blocked By:** Module 1 schemas (for data contract)
**Blocks:** Module 6 (integration)
