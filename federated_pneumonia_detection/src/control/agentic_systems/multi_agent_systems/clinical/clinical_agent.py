"""Clinical Interpretation Agent for X-ray predictions.

LLM-powered agent that generates clinical interpretations of pneumonia detection
results, including risk assessment, confidence explanation, and recommendations.

Follows the pattern from arxiv_agent/engine.py.
"""

import logging
from typing import Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


CLINICAL_SYSTEM_PROMPT = """You are an AI clinical assistant specialized in interpreting chest X-ray pneumonia detection results.

Your role is to provide clear, professional interpretations of AI model predictions for healthcare providers. You are NOT making diagnoses - you are explaining AI model outputs to help radiologists and clinicians make informed decisions.

Guidelines:
1. Be precise and clinical in your language
2. Always acknowledge model limitations and uncertainty
3. Emphasize the need for professional medical review
4. Highlight any factors that increase false-negative risk
5. Provide actionable recommendations based on confidence levels
6. Never claim certainty - use language like "suggests", "indicates", "consistent with"

Risk Level Guidelines:
- LOW: High confidence normal prediction (>90% confidence)
- MODERATE: Lower confidence predictions OR borderline probabilities
- HIGH: Positive pneumonia prediction with good confidence
- CRITICAL: High confidence pneumonia with additional concerning factors

False Negative Risk Assessment:
- Consider: low confidence, borderline probabilities, image quality issues
- Higher risk requires more urgent follow-up recommendations"""


class ClinicalAnalysisResponse(BaseModel):
    """Structured response with clinical interpretation."""
    summary: str = Field(description="Brief clinical summary of the finding")
    confidence_explanation: str = Field(description="Explanation of what the confidence level means")
    risk_level: str = Field(description="LOW, MODERATE, HIGH, or CRITICAL")
    false_negative_risk: str = Field(description="LOW, MODERATE, or HIGH")
    risk_factors: List[str] = Field(description="Factors contributing to the assessment")
    recommendations: List[str] = Field(description="Clinical recommendations")


class ClinicalInterpretationAgent:
    """Agent for generating clinical interpretations of predictions.

    Uses Gemini with structured output for reliable, parseable responses.
    """

    def __init__(self) -> None:
        """Initialize the clinical interpretation agent."""
        logger.info("[ClinicalAgent] Initializing...")

        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-pro-preview",
                temperature=0.3,
                max_tokens=1024,
            )
            self.structured_llm = self.llm.with_structured_output(ClinicalAnalysisResponse)
            logger.info("[ClinicalAgent] LLM initialized successfully")
        except Exception as e:
            logger.error(f"[ClinicalAgent] Failed to initialize: {e}", exc_info=True)
            raise

    async def interpret(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_probability: float,
        normal_probability: float,
        image_info: Optional[dict] = None,
    ) -> Optional[ClinicalAnalysisResponse]:
        """Generate clinical interpretation for a prediction.

        Args:
            predicted_class: NORMAL or PNEUMONIA
            confidence: Model confidence (0-1)
            pneumonia_probability: Probability of pneumonia (0-1)
            normal_probability: Probability of normal (0-1)
            image_info: Optional metadata about the input image.

        Returns:
            ClinicalAnalysisResponse or None if generation fails.
        """
        try:
            user_prompt = self._build_prompt(
                predicted_class, confidence, pneumonia_probability,
                normal_probability, image_info
            )

            messages = [
                SystemMessage(content=CLINICAL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response: ClinicalAnalysisResponse = await self.structured_llm.ainvoke(messages)
            return response

        except Exception as e:
            logger.error(f"[ClinicalAgent] Interpretation failed: {e}", exc_info=True)
            return None

    def _build_prompt(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_probability: float,
        normal_probability: float,
        image_info: Optional[dict],
    ) -> str:
        """Build the user prompt for clinical interpretation."""
        prompt = f"""Analyze this chest X-ray AI prediction and provide a clinical interpretation:

## Model Prediction Results
- **Predicted Class**: {predicted_class}
- **Model Confidence**: {confidence:.1%}
- **Pneumonia Probability**: {pneumonia_probability:.1%}
- **Normal Probability**: {normal_probability:.1%}

## Model Information
- Model: ResNet50-based pneumonia classifier
- Training: Federated learning with focal loss
- Validation Accuracy: 98.8%
"""

        if image_info:
            prompt += f"""
## Image Information
- Filename: {image_info.get('filename', 'Unknown')}
- Dimensions: {image_info.get('size', 'Unknown')}
"""

        prompt += """
Please provide:
1. A clinical summary of what this prediction means
2. An explanation of the confidence level and its implications
3. Risk assessment (risk_level, false_negative_risk, contributing factors)
4. Actionable recommendations for the clinical team

Remember: You are interpreting AI output, not making a diagnosis."""

        return prompt
