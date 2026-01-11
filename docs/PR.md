# Project Requirements Document

## 3.1 Functional Requirements (FR)

### 3.1.1 Dataset Upload (User-Facing Dashboard & Backend)

**Table 3.1.1: Dataset Upload Requirements**

| ID | Module | Version | Description |
|----|--------|---------|-------------|
| FR-DM-01 | Dashboard | v1 | The system shall allow a user to upload a dataset of chest X-ray images (e.g., in a structured format). |
| FR-DM-02 | Backend | v1 | The system shall validate the uploaded dataset for basic structure and image format compatibility. |
| FR-DM-03 | Backend | v1 | The system shall temporarily store or directly use the uploaded dataset for the current experiment session. |

---

### 3.1.2 Experiment Configuration (User-Facing Dashboard)

This section outlines what the underlying system, scripts, and tools used by the developer must be capable of to produce the data and models that feed into the dashboard.

**Table 3.1.2: Experiment Configuration Requirements**

| ID | Version | Description |
|----|---------|-------------|
| FR-CONF-01 | v1 | The system (Dashboard) shall allow the user to select the training mode: Centralized or Federated. |
| FR-CONF-02 | v1 | The system (Dashboard) shall allow the user to configure common model training hyperparameters for the fixed ResNet50 V2 model (using ImageNet pre-trained weights and a fixed preprocessing/augmentation pipeline), including: Number of fine-tune layers, Learning rate, Weight decay, Number of training epochs, Batch size. |
| FR-CONF-03 | v1 | If Federated Learning mode is selected, the system (Dashboard) shall allow the user to configure FL-specific parameters: Number of simulated clients, Number of federated rounds, Number of local epochs per client per round. |
| FR-CONF-04 | v1 | The system (Dashboard) shall allow the user to specify the train/validation split ratio for the uploaded dataset. |

---

### 3.1.3 Experiment Execution (User-Facing Dashboard & Backend)

**Table 3.1.3: Experiment Execution Requirements**

| ID | System | Version | Description |
|----|--------|---------|-------------|
| FR-EXEC-01 | Dashboard | v1 | Allow the user to initiate the training/simulation based on the current configuration and uploaded dataset. |
| FR-EXEC-02 | Backend | v1 | Execute the selected training mode (Centralized or Federated) using the fixed ResNet50 V2 model architecture, fixed preprocessing/augmentation pipeline, user-defined configurations, and the uploaded dataset. |
| FR-EXEC-03 | Backend | v1 | Perform the fixed data preprocessing (resizing to 224x224, normalization) and fixed data augmentation on the uploaded dataset according to the train/validation split. |
| FR-EXEC-04 | Backend | v1 | For Centralized mode, train the ResNet50 V2 model. |
| FR-EXEC-05 | Backend | v1 | For Federated mode: Partition data among clients, Instantiate clients (each training a local ResNet50 V2 model), Implement Federated Averaging (FedAvg), and Manage communication rounds and model updates. |
| FR-EXEC-06 | Backend | v1 | Handle model checkpointing (saving the best model for the current session). |
| FR-EXEC-07 | Dashboard | v1 | Provide feedback on experiment status (e.g., pending, running, completed, error). |
| FR-EXEC-08 | Dashboard | v1 | Display real-time or near real-time training progress. |

---

### 3.1.4 Model Evaluation (Backend - Automatic)

**Table 3.1.4: Model Evaluation Requirements**

| Requirement ID | Version | Description |
|----------------|---------|-------------|
| FR-EVAL-01 | v1 | The system (Backend) shall automatically evaluate the trained ResNet50 V2 model on the validation portion of the uploaded dataset after training completion. |
| FR-EVAL-02 | v1 | The system (Backend) shall compute and store (for the current session) standard performance metrics: accuracy, precision, recall, F1-score, and AUC. |
| FR-EVAL-03 | v1 | The system (Backend) shall generate and store (for the current session) a classification report. |
| FR-EVAL-04 | v1 | The system (Backend) shall generate and store (for the current session) a confusion matrix. |

---

### 3.1.5 Results Visualization (User-Facing Dashboard)

**Table 3.1.5: Results Visualization Requirements**

| ID | Version | Functional Requirement |
|----|---------|------------------------|
| FR-VIS-01 | v1 | The system (Dashboard) shall display the computed performance metrics for the completed experiment (current session). |
| FR-VIS-02 | v1 | The system (Dashboard) shall display the classification report for the completed experiment (current session). |
| FR-VIS-03 | v1 | The system (Dashboard) shall display the confusion matrix for the completed experiment (current session), potentially as a heatmap. |
| FR-VIS-04 | v1 | The system (Dashboard) shall display training progress curves if logged during training (current session). |
| FR-VIS-05 | v1 | The system (Dashboard) shall allow users to download generated reports and metrics for the current experiment. |

---

### 3.1.6 Inference Endpoints

**Table 3.1.6: Inference Endpoints Requirements**

| ID | Module | Version | Description |
|----|---------|---------|-------------|
| FR-INF-01 | InferenceEngine | v2 | The system shall provide a core inference engine capable of loading a pre-trained pneumonia detection model from a checkpoint file and performing binary classification (NORMAL or PNEUMONIA) on chest X-ray images. |
| FR-INF-02 | InferenceEngine | v2 | The system shall perform image preprocessing including resizing to 224x224 pixels, center cropping, tensor conversion, and normalization using ImageNet mean and standard deviation values before model inference. |
| FR-INF-03 | InferenceEngine | v2 | The system shall support automatic device detection and inference execution on GPU (CUDA) when available, with fallback to CPU when GPU is unavailable. |
| FR-INF-04 | InferenceEngine | v2 | The system shall return prediction results including predicted class, confidence score (0.0 to 1.0), pneumonia probability, and normal probability for each inference request. |
| FR-INF-05 | InferenceEngine | v2 | The system shall provide model information including model version, current device (CPU/GPU), GPU availability status, and checkpoint file path. |
| FR-INF-06 | InferenceService | v2 | The system shall implement a singleton service layer that manages the lifecycle of the inference engine and provides a unified interface for inference operations. |
| FR-INF-07 | InferenceService | v2 | The system shall support lazy loading of inference engine and clinical interpretation agent to initialize resources only when first requested. |
| FR-INF-08 | InferenceService | v2 | The system shall provide a health check endpoint that returns service status, model loaded status, GPU availability, and model version information. |
| FR-INF-09 | InferenceService | v2 | The system shall generate AI-powered clinical interpretations for predictions including summary, confidence explanation, risk assessment, and clinical recommendations when the clinical interpretation agent is available. |
| FR-INF-10 | InferenceService | v2 | The system shall provide a rule-based fallback clinical interpretation mechanism when the AI clinical agent is unavailable, including risk level assessment and clinical recommendations. |
| FR-INF-11 | InferenceUtils | v2 | The system shall support batch inference processing that accepts multiple images and returns individual results for each image along with aggregate statistics. |
| FR-INF-12 | InferenceUtils | v2 | The system shall compute batch summary statistics including total images processed, successful predictions, failed predictions, class distribution (NORMAL/PNEUMONIA counts), average confidence, average processing time, and high-risk count. |
| FR-INF-13 | InferenceSchemas | v2 | The system shall define a standardized InferenceResponse schema containing success flag, prediction results, optional clinical interpretation, model version, and processing time in milliseconds. |
| FR-INF-14 | InferenceSchemas | v2 | The system shall define a BatchInferenceResponse schema for batch inference operations containing per-image results, summary statistics, model version, and total processing time. |
| FR-INF-15 | InferenceSchemas | v2 | The system shall define a HealthCheckResponse schema with status indicator, model loaded flag, GPU availability flag, and model version. |
| FR-INF-16 | InferenceSchemas | v2 | The system shall define a SingleImageResult schema for tracking individual image results in batch operations including filename, success flag, prediction, clinical interpretation, error message, and processing time. |
| FR-INF-17 | InferenceSchemas | v2 | The system shall define an InferenceError schema for failed inference requests with success flag set to false, error type identifier, and human-readable error detail. |
| FR-INF-18 | InferenceSchemas | v2 | The system shall define ClinicalInterpretation schema with summary text, confidence explanation, RiskAssessment object, recommendations list, and medical disclaimer. |
| FR-INF-19 | InferenceSchemas | v2 | The system shall define RiskAssessment schema with risk level string (LOW, MODERATE, HIGH, CRITICAL), false negative risk assessment, and contributing factors list. |
| FR-INF-20 | InferenceUtils | v2 | The system shall implement risk assessment logic that evaluates confidence thresholds and predicted class to determine overall risk level and false negative risk for clinical interpretation. |

---

### 3.1.7 Agentic Systems

**Table 3.1.7: Agentic Systems Requirements**

| ID | Module | Version | Description |
|----|---------|---------|-------------|
| FR-AGENT-01 | ArxivAgent | v2 | The system shall provide a LangChain-based research assistant capable of searching Arxiv papers via MCP protocol and querying local RAG knowledge base. |
| FR-AGENT-02 | Query Classification | v2 | The system shall automatically classify user queries to determine whether tool-augmented "research" mode or conversational "basic" mode is appropriate. |
| FR-AGENT-03 | Streaming Response | v2 | The system shall support server-sent events (SSE) for token-by-token streaming of agent responses to the frontend. |
| FR-AGENT-04 | Conversation History | v2 | The system shall maintain session-based conversation history with configurable maximum turns to enable multi-turn dialogues. |
| FR-AGENT-05 | RAG Pipeline | v2 | The system shall provide a local Retrieval-Augmented Generation pipeline for processing PDF documents, semantic chunking, and storing embeddings in PostgreSQL. |
| FR-AGENT-06 | Vector Database | v2 | The system shall use PGVector with HuggingFace embeddings (all-MiniLM-L6-v2) for semantic search across uploaded research papers and medical documents. |
| FR-AGENT-07 | Clinical Interpretation Agent | v2 | The system shall provide a clinical interpretation agent that generates structured risk assessments (LOW/MODERATE/HIGH/CRITICAL) and recommendations based on model predictions. |
| FR-AGENT-08 | False Negative Risk Assessment | v2 | The system shall assess false negative risk based on prediction confidence, probability scores, and image quality indicators. |
| FR-AGENT-09 | MCP Manager | v2 | The system shall provide a singleton Model Context Protocol manager that manages the arxiv-mcp-server lifecycle and exposes tools as LangChain tools. |
| FR-AGENT-10 | Arxiv Embedding Tool | v2 | The system shall provide a tool for downloading Arxiv papers via MCP, chunking markdown content, and embedding papers into the knowledge base. |
| FR-AGENT-11 | Tool Orchestration | v2 | The system shall autonomously invoke appropriate tools (RAG or Arxiv) based on query analysis and emit real-time status updates for tool usage. |
| FR-AGENT-12 | Structured Output | v2 | The system shall generate structured, parseable responses with clinical interpretations including summary, confidence explanation, risk level, risk factors, and recommendations. |
| FR-AGENT-13 | Error Handling | v2 | The system shall implement graceful fallbacks for LLM failures and return safe, structured error responses instead of crashing the system. |
| FR-AGENT-14 | Prompt Engineering | v2 | The system shall store prompts in dedicated template files and never hardcode prompts inside logic functions. |
| FR-AGENT-15 | Async Execution | v2 | The system shall use async/await for all LLM calls to prevent blocking the FastAPI event loop. |
| FR-AGENT-16 | LLM Observability | v2 | The system shall integrate LangSmith tracing for all chains and agents with full input/output logging, token tracking, and automated evaluation (hallucination detection, answer relevance). |

---

### 3.1.8 System Monitoring & Analytics

**Table 3.1.8: System Monitoring & Analytics Requirements**

| ID | Module | Version | Description |
|----|--------|---------|-------------|
| FR-MON-01 | Dashboard | v2 | The system shall provide real-time monitoring of training progress through WebSocket connections, broadcasting metrics (e.g., loss, accuracy, precision, recall) per epoch or per federated round to all connected clients. |
| FR-MON-02 | Runs API | v2 | The system shall provide an endpoint to list all training runs with pagination, filtering by status and training mode, and sorting capabilities for run tracking and history management. |
| FR-MON-03 | Runs API | v2 | The system shall provide an endpoint to retrieve complete training results for a specific run, including final metrics, training history, metadata, and confusion matrix. |
| FR-MON-04 | Runs API | v2 | The system shall provide an endpoint to retrieve per-round metrics for federated learning runs, including global performance metrics and per-client training metrics for chart-ready visualization. |
| FR-MON-05 | Runs API | v2 | The system shall provide an endpoint to retrieve detailed server-side evaluation metrics for federated training, including per-round evaluations and summary statistics (best, average, worst metrics). |
| FR-MON-06 | Runs API | v2 | The system shall provide an endpoint to download training results in multiple formats (CSV, JSON, text summary) for external analysis and archival purposes. |
| FR-MON-07 | Runs API | v2 | The system shall provide an analytics summary endpoint that aggregates statistics across all training runs, including separate metrics for centralized and federated modes, run counts, success rates, and top-performing runs by specified criteria. |
| FR-MON-08 | Logging | v2 | The system shall implement structured logging with correlation ID tracking to enable end-to-end tracing of requests and operations across all layers. |
| FR-MON-09 | Logging | v2 | The system shall support multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) with configurable formatting for different debugging and operational needs. |
| FR-MON-10 | Logging | v2 | The system shall provide a centralized logger utility that creates and configures named loggers with consistent format and handlers across the entire application. |

---

## 3.2 Non-Functional Requirements (NFR)

Non-functional requirements specify the quality attributes and constraints under which the system must operate. For a medical AI system like the Federated Learning system for Pneumonia Detection, these requirements are crucial for ensuring its effectiveness, trustworthiness, and practical deployability, beyond merely its functional capabilities. They define how well the system performs, its robustness, and its adherence to critical considerations like data privacy and security.

### 3.2.1 Usability

**Table 3.2.1.1: Usability Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-01 | The dashboard interface shall be intuitive and accessible to users with varying technical backgrounds, allowing them to configure and run experiments with minimal prior instruction. |
| NFR-02 | All visualizations (charts, graphs, tables) presented on the dashboard must be clearly labeled with appropriate titles, axis labels, and legends, ensuring that data insights are easily interpretable. |
| NFR-03 | Users shall be able to easily configure all common model training parameters and FL-specific parameters through clearly presented input fields and options within the dashboard interface. |

---

### 3.2.2 Performance

**Table 3.2.1.2: Performance Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-04 | The dashboard interface shall remain responsive during user interactions (e.g., configuring parameters, navigating views) with typical response times of less than 2 seconds. For longer operations (e.g., starting training), visual feedback (e.g., loading indicators, progress bars) shall be provided to the user. |
| NFR-05 | The centralized model training on the 10% RSNA sample dataset shall complete within a reasonable timeframe (e.g., X hours) on the specified development hardware (e.g., a single NVIDIA V100 GPU), to facilitate rapid iteration. |
| NFR-06 | The federated learning simulation (FedAvg with 2 clients) on the 10% RSNA sample dataset shall execute efficiently, completing per federated round within a predictable timeframe (e.g., Y minutes per round), with execution time scaling predictably with the number of clients and rounds. |

---

### 3.2.3 Scalability

**Table 3.2.1.3: Scalability Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-07 | The system's architecture shall be designed to allow for the future expansion to support a larger number of simulated clients (e.g., up to 10-20 clients) and larger dataset fractions or the full RSNA dataset, with a clear understanding of expected resource scaling. |

---

### 3.2.4 Security & Privacy

**Table 3.2.1.4: Security & Privacy Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-08 | The system shall ensure that raw patient X-ray image data is never transferred from the simulated client environments to the central server or other clients. Only aggregated model parameters shall be exchanged, upholding the core principle of Federated Learning. |
| NFR-09 | All communication of model updates between simulated clients and the central server (via the Flower framework) shall utilize secure, encrypted channels (e.g., TLS/SSL) to protect against unauthorized interception. |
| NFR-10 | Access to the interactive dashboard and its functionalities (e.g., experiment configuration, results viewing) shall be restricted to authorized personnel. For the prototype, this implies local machine access; future versions may incorporate explicit authentication. |

---

### 3.2.5 Reliability

**Table 3.2.1.5: Reliability Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-11 | The calculated and displayed performance metrics (e.g., Accuracy, Precision, Recall, F1-Score, AUC) for both centralized and federated models must be accurate and consistent with the underlying model computations and standard evaluation methodologies. |
| NFR-12 | The system shall provide informative and actionable error messages if an experiment fails or if invalid configuration parameters are entered by the user, guiding them towards troubleshooting and correction. |
| NFR-13 | The system shall be resilient to temporary disconnections or failures of individual simulated clients during federated learning, allowing the global training to continue if a minimum number of clients are available, and logging such events. |
| NFR-14 | The system shall implement robust model checkpointing mechanisms to save the best performing model weights during training, enabling reproducibility and recovery from unexpected interruptions. |

---

### 3.2.6 Maintainability

**Table 3.2.1.6: Maintainability Requirements**

| Req ID | Description |
|--------|-------------|
| NFR-15 | The system's codebase shall adhere to a modular design principle, allowing for future enhancements such as the integration of new federated aggregation algorithms (e.g., FedProx), different deep learning architectures, or additional evaluation metrics with minimal impact on existing components. |
| NFR-16 | The system shall be developed using widely supported and actively maintained open-source libraries and frameworks (e.g., PyTorch, Flower, Streamlit) to ensure long-term support, community resources, and ease of future upgrades.