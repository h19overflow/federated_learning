from typing import List, Tuple
from pathlib import Path
from PIL import Image
import random


class DatasetLoader:
    """Load test images for benchmarking."""

    def __init__(self, image_dir: str, max_images: int = 1000):
        self.image_dir = Path(image_dir)
        self.max_images = max_images
        self.images = self._discover_images()

    def _discover_images(self) -> List[Path]:
        extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = []
        for ext in extensions:
            images.extend(self.image_dir.rglob(f'*{ext}'))
        random.shuffle(images)
        return images[:self.max_images]

    def load_images(self, count: int = None) -> List[Image.Image]:
        count = count or len(self.images)
        loaded = []
        for img_path in self.images[:count]:
            try:
                img = Image.open(img_path).convert('RGB')
                loaded.append(img)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
        return loaded

    def get_sample_paths(self, count: int = None) -> List[Path]:
        return self.images[:count or len(self.images)]
