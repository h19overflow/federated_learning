"""Clinical interpretation component."""

import logging

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
)

logger = logging.getLogger(__name__)


class ClinicalInterpreter:
    """Generates clinical interpretations from predictions."""

    def __init__(self, clinical_agent=None):
        """Initialize with optional clinical agent."""
        self._agent = clinical_agent

    def set_agent(self, agent):
        """Set the clinical agent (for lazy loading)."""
        self._agent = agent

    async def generate(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_prob: float,
        normal_prob: float,
        prediction: InferencePrediction,
        image_info: dict,
    ) -> ClinicalInterpretation:
        """Generate clinical interpretation using agent or fallback."""
        if self._agent:
            try:
                agent_response = await self._agent.interpret(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    pneumonia_probability=pneumonia_prob,
                    normal_probability=normal_prob,
                    image_info=image_info,
                )
                if agent_response:
                    return ClinicalInterpretation(
                        summary=agent_response.summary,
                        confidence_explanation=agent_response.confidence_explanation,
                        risk_assessment=RiskAssessment(
                            risk_level=agent_response.risk_level,
                            false_negative_risk=agent_response.false_negative_risk,
                            factors=agent_response.risk_factors,
                        ),
                        recommendations=agent_response.recommendations,
                    )
            except Exception as e:
                logger.warning(f"Clinical agent failed, using fallback: {e}")

        return self._generate_fallback(prediction)

    def _generate_fallback(
        self,
        prediction: InferencePrediction,
    ) -> ClinicalInterpretation:
        """Generate rule-based clinical interpretation."""
        # Determine risk level
        if prediction.predicted_class == PredictionClass.PNEUMONIA:
            if prediction.confidence >= 0.9:
                risk_level, fn_risk = "HIGH", "LOW"
            elif prediction.confidence >= 0.7:
                risk_level, fn_risk = "MODERATE", "LOW"
            else:
                risk_level, fn_risk = "MODERATE", "MODERATE"
        else:
            if prediction.confidence >= 0.9:
                risk_level, fn_risk = "LOW", "LOW"
            elif prediction.confidence >= 0.7:
                risk_level, fn_risk = "LOW", "MODERATE"
            else:
                risk_level, fn_risk = "MODERATE", "HIGH"

        # Build factors
        factors = []
        if prediction.confidence < 0.7:
            factors.append("Low model confidence suggests uncertainty")
        if (
            prediction.predicted_class == PredictionClass.NORMAL
            and prediction.pneumonia_probability > 0.3
        ):
            factors.append("Elevated pneumonia probability warrants review")
        if prediction.confidence >= 0.9:
            factors.append("High confidence from validated model")

        # Build recommendations
        recommendations = []
        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.extend(
                [
                    "Immediate radiologist review recommended",
                    "Consider clinical correlation with symptoms",
                ],
            )
        elif risk_level == "MODERATE":
            recommendations.append("Radiologist review within 24 hours recommended")
        else:
            recommendations.append("Standard review workflow appropriate")

        if fn_risk in ["MODERATE", "HIGH"]:
            recommendations.append(
                "Consider repeat imaging if clinical suspicion persists",
            )

        # Build summary
        if prediction.predicted_class == PredictionClass.PNEUMONIA:
            summary = (
                f"Model detects signs consistent with pneumonia with "
                f"{prediction.confidence:.1%} confidence."
            )
        else:
            summary = (
                f"No definitive signs of pneumonia detected. "
                f"Model confidence: {prediction.confidence:.1%}."
            )

        # Confidence explanation
        if prediction.confidence >= 0.9:
            conf_exp = "High confidence prediction."
        elif prediction.confidence >= 0.7:
            conf_exp = "Moderate confidence; radiologist review advised."
        else:
            conf_exp = "Lower confidence; expert review recommended."

        return ClinicalInterpretation(
            summary=summary,
            confidence_explanation=conf_exp,
            risk_assessment=RiskAssessment(
                risk_level=risk_level,
                false_negative_risk=fn_risk,
                factors=factors,
            ),
            recommendations=recommendations,
        )
