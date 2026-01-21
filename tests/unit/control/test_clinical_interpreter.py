"""
Unit tests for ClinicalInterpreter component.
Tests clinical interpretation generation and fallback logic.
"""

import pytest

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
)
from federated_pneumonia_detection.src.control.model_inferance.internals.clinical_interpreter import (
    ClinicalInterpreter,
)


class TestClinicalInterpreter:
    """Tests for ClinicalInterpreter class."""

    @pytest.fixture
    def interpreter(self):
        """Create ClinicalInterpreter instance."""
        return ClinicalInterpreter()

    @pytest.fixture
    def interpreter_with_agent(self, mock_clinical_agent):
        """Create ClinicalInterpreter with mock agent."""
        return ClinicalInterpreter(clinical_agent=mock_clinical_agent)

    # =========================================================================
    # Test initialization
    # =========================================================================

    def test_init_without_agent(self, interpreter):
        """Test initialization without agent."""
        assert interpreter._agent is None

    def test_init_with_agent(self, mock_clinical_agent):
        """Test initialization with agent."""
        interpreter = ClinicalInterpreter(clinical_agent=mock_clinical_agent)
        assert interpreter._agent is mock_clinical_agent

    def test_set_agent(self, interpreter, mock_clinical_agent):
        """Test set_agent method."""
        assert interpreter._agent is None
        interpreter.set_agent(mock_clinical_agent)
        assert interpreter._agent is mock_clinical_agent

    # =========================================================================
    # Test generate with agent
    # =========================================================================

    @pytest.mark.asyncio
    async def test_generate_with_agent(
        self,
        interpreter_with_agent,
        sample_pneumonia_prediction,
    ):
        """Test generate calls agent and returns interpretation."""
        image_info = {"filename": "test.jpg", "size": (512, 512)}

        interpretation = await interpreter_with_agent.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=sample_pneumonia_prediction,
            image_info=image_info,
        )

        assert isinstance(interpretation, ClinicalInterpretation)
        assert "Agent interpretation" in interpretation.summary

        # Agent should have been called
        interpreter_with_agent._agent.interpret.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_agent_normal(
        self,
        interpreter_with_agent,
        sample_normal_prediction,
    ):
        """Test generate with NORMAL prediction."""
        image_info = {"filename": "test.jpg", "size": (512, 512)}

        interpretation = await interpreter_with_agent.generate(
            predicted_class="NORMAL",
            confidence=0.88,
            pneumonia_prob=0.12,
            normal_prob=0.88,
            prediction=sample_normal_prediction,
            image_info=image_info,
        )

        assert isinstance(interpretation, ClinicalInterpretation)

    # =========================================================================
    # Test generate with failing agent
    # =========================================================================

    @pytest.mark.asyncio
    async def test_generate_failing_agent_uses_fallback(
        self,
        sample_pneumonia_prediction,
    ):
        """Test that failing agent falls back to rule-based interpretation."""
        failing_agent = pytest.mock.Mock()
        failing_agent.interpret = pytest.mock.AsyncMock(
            side_effect=RuntimeError("Agent failed"),
        )

        interpreter = ClinicalInterpreter(clinical_agent=failing_agent)

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=sample_pneumonia_prediction,
            image_info={"filename": "test.jpg"},
        )

        assert isinstance(interpretation, ClinicalInterpretation)
        # Should use fallback, not agent response
        assert "pneumonia" in interpretation.summary.lower()

    @pytest.mark.asyncio
    async def test_generate_with_none_agent(
        self,
        interpreter,
        sample_pneumonia_prediction,
    ):
        """Test generate with no agent uses fallback."""
        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=sample_pneumonia_prediction,
            image_info={"filename": "test.jpg"},
        )

        assert isinstance(interpretation, ClinicalInterpretation)
        assert "pneumonia" in interpretation.summary.lower()

    # =========================================================================
    # Test fallback interpretation - PNEUMONIA cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_fallback_pneumonia_high_confidence(
        self,
        interpreter,
        sample_pneumonia_prediction,
    ):
        """Test fallback for PNEUMONIA with high confidence."""
        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.95,
            pneumonia_prob=0.95,
            normal_prob=0.05,
            prediction=sample_pneumonia_prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "HIGH"
        assert interpretation.risk_assessment.false_negative_risk == "LOW"
        assert "High confidence" in interpretation.confidence_explanation

    @pytest.mark.asyncio
    async def test_fallback_pneumonia_moderate_confidence(self, interpreter):
        """Test fallback for PNEUMONIA with moderate confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.75,
            pneumonia_probability=0.75,
            normal_probability=0.25,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.75,
            pneumonia_prob=0.75,
            normal_prob=0.25,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "MODERATE"
        assert interpretation.risk_assessment.false_negative_risk == "LOW"

    @pytest.mark.asyncio
    async def test_fallback_pneumonia_low_confidence(self, interpreter):
        """Test fallback for PNEUMONIA with low confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.55,
            pneumonia_probability=0.55,
            normal_probability=0.45,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.55,
            pneumonia_prob=0.55,
            normal_prob=0.45,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "MODERATE"
        assert interpretation.risk_assessment.false_negative_risk == "MODERATE"
        assert "uncertainty" in " ".join(interpretation.risk_assessment.factors).lower()

    # =========================================================================
    # Test fallback interpretation - NORMAL cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_fallback_normal_high_confidence(
        self,
        interpreter,
        sample_normal_prediction,
    ):
        """Test fallback for NORMAL with high confidence."""
        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.95,
            pneumonia_prob=0.05,
            normal_prob=0.95,
            prediction=sample_normal_prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "LOW"
        assert interpretation.risk_assessment.false_negative_risk == "LOW"

    @pytest.mark.asyncio
    async def test_fallback_normal_moderate_confidence(self, interpreter):
        """Test fallback for NORMAL with moderate confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.75,
            pneumonia_probability=0.25,
            normal_probability=0.75,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.75,
            pneumonia_prob=0.25,
            normal_prob=0.75,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "LOW"
        assert interpretation.risk_assessment.false_negative_risk == "MODERATE"

    @pytest.mark.asyncio
    async def test_fallback_normal_low_confidence(self, interpreter):
        """Test fallback for NORMAL with low confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.55,
            pneumonia_probability=0.45,
            normal_probability=0.55,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.55,
            pneumonia_prob=0.45,
            normal_prob=0.55,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "MODERATE"
        assert interpretation.risk_assessment.false_negative_risk == "HIGH"

    # =========================================================================
    # Test boundary cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_fallback_boundary_0_7(self, interpreter):
        """Test fallback at confidence boundary 0.7."""
        # PNEUMONIA at exactly 0.7
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.70,
            pneumonia_probability=0.70,
            normal_probability=0.30,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.70,
            pneumonia_prob=0.70,
            normal_prob=0.30,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "MODERATE"

    @pytest.mark.asyncio
    async def test_fallback_boundary_0_9(self, interpreter):
        """Test fallback at confidence boundary 0.9."""
        # NORMAL at exactly 0.9
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.90,
            pneumonia_probability=0.10,
            normal_probability=0.90,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.90,
            pneumonia_prob=0.10,
            normal_prob=0.90,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "LOW"

    @pytest.mark.asyncio
    async def test_fallback_just_above_0_9(self, interpreter):
        """Test fallback just above 0.9."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.91,
            pneumonia_probability=0.09,
            normal_probability=0.91,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.91,
            pneumonia_prob=0.09,
            normal_prob=0.91,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.risk_assessment.risk_level == "LOW"

    # =========================================================================
    # Test factors generation
    # =========================================================================

    @pytest.mark.asyncio
    async def test_factors_low_confidence(self, interpreter):
        """Test low confidence factor is added."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.60,
            pneumonia_probability=0.60,
            normal_probability=0.40,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.60,
            pneumonia_prob=0.60,
            normal_prob=0.40,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        factors = interpretation.risk_assessment.factors
        assert any(
            "uncertainty" in f.lower() or "confidence" in f.lower() for f in factors
        )

    @pytest.mark.asyncio
    async def test_factors_high_confidence(self, interpreter):
        """Test high confidence factor is added."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.95,
            pneumonia_probability=0.95,
            normal_probability=0.05,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.95,
            pneumonia_prob=0.95,
            normal_prob=0.05,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        factors = interpretation.risk_assessment.factors
        assert any("high confidence" in f.lower() for f in factors)

    @pytest.mark.asyncio
    async def test_factors_elevated_pneumonia_prob(self, interpreter):
        """Test elevated pneumonia probability factor for NORMAL."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.60,
            pneumonia_probability=0.40,  # Elevated for NORMAL prediction
            normal_probability=0.60,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.60,
            pneumonia_prob=0.40,
            normal_prob=0.60,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        factors = interpretation.risk_assessment.factors
        assert any("pneumonia" in f.lower() and "review" in f.lower() for f in factors)

    # =========================================================================
    # Test recommendations generation
    # =========================================================================

    @pytest.mark.asyncio
    async def test_recommendations_high_risk(self, interpreter):
        """Test recommendations for high risk."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.95,
            pneumonia_probability=0.95,
            normal_probability=0.05,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.95,
            pneumonia_prob=0.95,
            normal_prob=0.05,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        recommendations = interpretation.recommendations
        assert len(recommendations) >= 2
        assert any(
            "immediate" in r.lower() or "radiologist" in r.lower()
            for r in recommendations
        )

    @pytest.mark.asyncio
    async def test_recommendations_moderate_risk(self, interpreter):
        """Test recommendations for moderate risk."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.75,
            pneumonia_probability=0.75,
            normal_probability=0.25,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.75,
            pneumonia_prob=0.75,
            normal_prob=0.25,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        recommendations = interpretation.recommendations
        assert any(
            "24 hours" in r.lower() or "review" in r.lower() for r in recommendations
        )

    @pytest.mark.asyncio
    async def test_recommendations_low_risk(self, interpreter):
        """Test recommendations for low risk."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.95,
            pneumonia_probability=0.05,
            normal_probability=0.95,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.95,
            pneumonia_prob=0.05,
            normal_prob=0.95,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        recommendations = interpretation.recommendations
        assert any(
            "standard" in r.lower() or "workflow" in r.lower() for r in recommendations
        )

    @pytest.mark.asyncio
    async def test_recommendations_high_false_negative_risk(self, interpreter):
        """Test recommendations include repeat imaging for high FN risk."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.55,
            pneumonia_probability=0.45,
            normal_probability=0.55,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.55,
            pneumonia_prob=0.45,
            normal_prob=0.55,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        recommendations = interpretation.recommendations
        assert any(
            "repeat" in r.lower() or "imaging" in r.lower() for r in recommendations
        )

    # =========================================================================
    # Test summary generation
    # =========================================================================

    @pytest.mark.asyncio
    async def test_summary_pneumonia(self, interpreter):
        """Test summary for PNEUMONIA."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.85,
            pneumonia_probability=0.85,
            normal_probability=0.15,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.85,
            pneumonia_prob=0.85,
            normal_prob=0.15,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert "pneumonia" in interpretation.summary.lower()
        assert "85.0%" in interpretation.summary or "85%" in interpretation.summary

    @pytest.mark.asyncio
    async def test_summary_normal(self, interpreter):
        """Test summary for NORMAL."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.NORMAL,
            confidence=0.85,
            pneumonia_probability=0.15,
            normal_probability=0.85,
        )

        interpretation = await interpreter.generate(
            predicted_class="NORMAL",
            confidence=0.85,
            pneumonia_prob=0.15,
            normal_prob=0.85,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert (
            "no definitive" in interpretation.summary.lower()
            or "normal" in interpretation.summary.lower()
        )
        assert "85.0%" in interpretation.summary or "85%" in interpretation.summary

    # =========================================================================
    # Test confidence explanation
    # =========================================================================

    @pytest.mark.asyncio
    async def test_confidence_explanation_high(self, interpreter):
        """Test confidence explanation for high confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.92,
            pneumonia_probability=0.92,
            normal_probability=0.08,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert "high confidence" in interpretation.confidence_explanation.lower()

    @pytest.mark.asyncio
    async def test_confidence_explanation_moderate(self, interpreter):
        """Test confidence explanation for moderate confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.75,
            pneumonia_probability=0.75,
            normal_probability=0.25,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.75,
            pneumonia_prob=0.75,
            normal_prob=0.25,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert "moderate" in interpretation.confidence_explanation.lower()

    @pytest.mark.asyncio
    async def test_confidence_explanation_low(self, interpreter):
        """Test confidence explanation for low confidence."""
        prediction = InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.55,
            pneumonia_probability=0.55,
            normal_probability=0.45,
        )

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.55,
            pneumonia_prob=0.55,
            normal_prob=0.45,
            prediction=prediction,
            image_info={"filename": "test.jpg"},
        )

        assert (
            "lower confidence" in interpretation.confidence_explanation.lower()
            or "expert review" in interpretation.confidence_explanation.lower()
        )

    # =========================================================================
    # Test image info handling
    # =========================================================================

    @pytest.mark.asyncio
    async def test_generate_with_image_info(
        self,
        interpreter,
        sample_pneumonia_prediction,
    ):
        """Test generate with image info parameter."""
        image_info = {
            "filename": "patient_123_xray.jpg",
            "size": (512, 512),
            "format": "JPEG",
        }

        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=sample_pneumonia_prediction,
            image_info=image_info,
        )

        # Image info is passed to agent, but fallback doesn't use it
        assert isinstance(interpretation, ClinicalInterpretation)

    # =========================================================================
    # Test disclaimer
    # =========================================================================

    @pytest.mark.asyncio
    async def test_interpretation_has_disclaimer(
        self,
        interpreter,
        sample_pneumonia_prediction,
    ):
        """Test interpretation includes disclaimer."""
        interpretation = await interpreter.generate(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
            prediction=sample_pneumonia_prediction,
            image_info={"filename": "test.jpg"},
        )

        assert interpretation.disclaimer is not None
        assert len(interpretation.disclaimer) > 0
        assert (
            "ai" in interpretation.disclaimer.lower()
            or "artificial" in interpretation.disclaimer.lower()
        )
