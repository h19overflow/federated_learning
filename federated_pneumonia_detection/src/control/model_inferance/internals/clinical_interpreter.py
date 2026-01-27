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
        # Determine risk level using dictionary dispatch
        risk_level, fn_risk = self._determine_risk_level(prediction)

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

        # Build recommendations using dictionary dispatch
        recommendations = self._build_recommendations(risk_level, fn_risk)

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

        # Confidence explanation using dictionary dispatch
        conf_exp = self._get_confidence_explanation(prediction.confidence)

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

    def _determine_risk_level(
        self,
        prediction: InferencePrediction,
    ) -> tuple[str, str]:
        """Determine risk level and false negative risk.

        Based on prediction class and confidence.
        Uses dictionary dispatch pattern to map (class, confidence_range)
        to (risk_level, fn_risk).

        Args:
            prediction: The inference prediction

        Returns:
            Tuple of (risk_level, false_negative_risk)
        """
        # Define risk mappings for each prediction class
        risk_mappings = {
            PredictionClass.PNEUMONIA: {
                "high": ("HIGH", "LOW"),  # confidence >= 0.9
                "moderate": ("MODERATE", "LOW"),  # confidence >= 0.7
                "low": ("MODERATE", "MODERATE"),  # confidence < 0.7
            },
            PredictionClass.NORMAL: {
                "high": ("LOW", "LOW"),  # confidence >= 0.9
                "moderate": ("LOW", "MODERATE"),  # confidence >= 0.7
                "low": ("MODERATE", "HIGH"),  # confidence < 0.7
            },
        }

        # Determine confidence range
        if prediction.confidence >= 0.9:
            confidence_range = "high"
        elif prediction.confidence >= 0.7:
            confidence_range = "moderate"
        else:
            confidence_range = "low"

        # Dispatch to correct risk mapping
        class_mapping = risk_mappings.get(prediction.predicted_class)
        if not class_mapping:
            # Fallback for unknown class
            return "MODERATE", "MODERATE"

        return class_mapping[confidence_range]

    def _build_recommendations(self, risk_level: str, fn_risk: str) -> list[str]:
        """Build recommendations based on risk levels using dictionary dispatch.

        Args:
            risk_level: The risk level (HIGH, MODERATE, LOW, CRITICAL)
            fn_risk: The false negative risk (HIGH, MODERATE, LOW)

        Returns:
            List of recommendations
        """
        # Map risk level to base recommendations
        risk_recommendations = {
            "HIGH": [
                "Immediate radiologist review recommended",
                "Consider clinical correlation with symptoms",
            ],
            "CRITICAL": [
                "Immediate radiologist review recommended",
                "Consider clinical correlation with symptoms",
            ],
            "MODERATE": ["Radiologist review within 24 hours recommended"],
            "LOW": ["Standard review workflow appropriate"],
        }

        recommendations = risk_recommendations.get(risk_level, [])

        # Add false negative risk recommendation
        if fn_risk in ["MODERATE", "HIGH"]:
            recommendations.append(
                "Consider repeat imaging if clinical suspicion persists",
            )

        return recommendations

    def _get_confidence_explanation(self, confidence: float) -> str:
        """Get confidence explanation based on confidence score.

        Uses dictionary dispatch pattern.

        Args:
            confidence: The confidence score (0.0 to 1.0)

        Returns:
            Confidence explanation string
        """
        # Map confidence ranges to explanations
        if confidence >= 0.9:
            return "High confidence prediction."
        elif confidence >= 0.7:
            return "Moderate confidence; radiologist review advised."
        else:
            return "Lower confidence; expert review recommended."
