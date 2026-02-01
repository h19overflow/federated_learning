# Inference API - Health Check Flow

**API**: `GET /api/inference/health`
**Entry**: `health_endpoints.py:22-36` → `InferenceService.get_info()`

---

## Step 1: Service Status Retrieval

**Files**:
- `health_endpoints.py` (lines 22-36)
- `inference_service.py` (lines 252-267)
- `inference_engine.py` (lines 127-139)

```mermaid
sequenceDiagram
    participant C as Client/Monitor
    participant API as HealthEndpoint
    participant S as InferenceService
    participant E as InferenceEngine

    C->>API: GET /api/inference/health
    API->>S: get_info()
    alt Engine not initialized
        S->>S: self.engine is None
        S-->>API: {
        S-->>API:   "status": "unhealthy",
        S-->>API:   "model_loaded": False,
        S-->>API:   "gpu_available": False,
        S-->>API:   "model_version": None
        S-->>API: }
    else Engine available
        S->>E: get_info()
        E->>E: Collect device info
        E-->>S: engine_info dict
        S-->>API: {
        S-->>API:   "status": "healthy",
        S-->>API:   "model_loaded": True,
        S-->>API:   "gpu_available": info["gpu_available"],
        S-->>API:   "model_version": info["model_version"]
        S-->>API: }
    end
    API->>API: HealthCheckResponse(**info)
    API-->>C: JSON Response
```

**Key Code**:
```python
# health_endpoints.py:22-36
@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service),
) -> HealthCheckResponse:
    info = service.get_info()
    return HealthCheckResponse(
        status=info["status"],
        model_loaded=info["model_loaded"],
        gpu_available=info.get("gpu_available", False),
        model_version=info.get("model_version"),
    )

# inference_service.py:252-267
def get_info(self) -> dict:
    if self.engine is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "gpu_available": False,
            "model_version": None,
        }
    info = self.engine.get_info()
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": info.get("gpu_available", False),
        "model_version": info.get("model_version"),
    }
```

---

## Step 2: Engine Information Collection

**Files**:
- `inference_engine.py` (lines 127-139)

```mermaid
sequenceDiagram
    participant E as InferenceEngine
    participant T as torch

    E->>E: get_info()
    E->>T: cuda.is_available()
    T-->>E: gpu_available: bool
    E->>E: self.model_version
    E->>E: self.device
    E->>E: self.checkpoint_path
    E-->>E: {
    E-->>E:   "model_version": self.model_version,
    E-->>E:   "device": self.device,
    E-->>E:   "gpu_available": torch.cuda.is_available(),
    E-->>E:   "checkpoint_path": str(self.checkpoint_path)
    E-->>E: }
```

**Key Code**:
```python
# inference_engine.py:132-139
def get_info(self) -> dict:
    return {
        "model_version": self.model_version,
        "device": self.device,
        "gpu_available": torch.cuda.is_available(),
        "checkpoint_path": str(self.checkpoint_path),
    }
```

---

## File Reference

| Layer | File | Key Lines | Purpose |
|-------|------|-----------|---------|
| **API** | `health_endpoints.py` | 22-36 | Health check route |
| **Control** | `inference_service.py` | 252-267 | Service health info |
| **Control** | `inference_engine.py` | 127-139 | Engine health info |
| **Schema** | `inference_schemas.py` | 160-173 | HealthCheckResponse |

---

## Health Status States

| Status | model_loaded | gpu_available | Meaning |
|--------|--------------|---------------|---------|
| `healthy` | True | True/False | Model loaded, ready for inference |
| `unhealthy` | False | False | Model not loaded, service unavailable |
| `degraded` | (Future) | - | Model loaded but with warnings |

---

## Response Schema

```python
# inference_schemas.py:160-173
class HealthCheckResponse(BaseModel):
    status: str                    # "healthy" | "unhealthy" | "degraded"
    model_loaded: bool             # Is model in memory
    gpu_available: bool            # Is CUDA available
    model_version: Optional[str]   # Checkpoint filename stem
```
