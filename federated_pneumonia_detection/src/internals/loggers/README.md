# Loggers Module

## Purpose
Provides a centralized logging configuration to ensure consistent log formatting and to silence noisy third-party libraries.

## Key Files
- `logging_config.py`: Main configuration logic. Silences `langchain_google_genai`, `google.genai`, and `uvicorn.access`.
- `logger.py`: Utility functions for retrieving configured loggers.

## How It Works
The `configure_logging()` function should be called once at application startup. It sets the root log level based on the `LOG_LEVEL` environment variable (defaulting to `INFO`) and applies specific overrides to third-party loggers.

## Usage
To get a logger in any module:

```python
from federated_pneumonia_detection.src.internals.loggers import get_logger

logger = get_logger(__name__)
logger.info("Operation started")
```

## Silenced Libraries
The following libraries are set to `WARNING` level by default to reduce noise:
- `langchain_google_genai`
- `google.genai`
- `uvicorn.access`
- `uvicorn` (set to `INFO`)
