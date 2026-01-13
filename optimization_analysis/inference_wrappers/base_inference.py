from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image


class BaseInferenceWrapper(ABC):
    """Abstract base class for inference implementations."""

    def __init__(self, name: str, checkpoint_path: str):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image):
        pass

    @abstractmethod
    def extract_features(self, preprocessed_data):
        pass

    @abstractmethod
    def classify(self, features):
        pass

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        preprocessed = self.preprocess(image)
        features = self.extract_features(preprocessed)
        return self.classify(features)

    def get_name(self) -> str:
        return self.name
