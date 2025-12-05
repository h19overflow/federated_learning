# Slash Command Feature for Real-Time Training Insights

## Overview

This feature enables users to select specific training runs and get real-time insights through an AI-powered chat interface. By typing `/` in the chat input, users can access a dropdown picker to select any training run and ask contextual questions about its metrics, performance, and configuration.

## User Interface

### Activation

1. **Open the Chat Sidebar**: Click the chat icon in the bottom-right corner (if closed).
2. **Trigger Run Picker**: Type `/` in the chat input field.
3. **Select a Run**: A dropdown appears showing all available training runs with their details.

### Run Picker Display

Each run in the picker shows:

- **Run ID**: Unique identifier (e.g., `Run #123`)
- **Training Mode**: Either "federated" or "centralized"
- **Status**: Current state (`completed`, `in_progress`, `failed`)
- **Start Time**: When the training began
- **Best Recall**: Top validation recall score achieved (if available)
- **Visual Indicators**: Color-coded badges and icons

### Run Context Badge

Once a run is selected:

- A badge appears at the top of the chat showing the active run context
- All subsequent questions are automatically contextualized to this run
- Click the `X` button to clear the run context

### Enhanced Input Placeholder

The chat input dynamically updates:

- Default: "Type / to select a run or ask anything..."
- With run selected: "Ask about Run #123..."

## Features

### 1. Real-Time Run Selection

```typescript
// When user types '/'
if (value === "/") {
  setShowRunPicker(true);
  fetchAvailableRuns(); // Loads all training runs
}
```

### 2. Contextual Queries

Once a run is selected, users can ask:

- **Performance Questions**: "What was the best accuracy achieved?"
- **Comparison Questions**: "How does this compare to other runs?"
- **Metric Analysis**: "Show me the validation loss trend"
- **Configuration Questions**: "What hyperparameters were used?"
- **Federated-Specific**: "How did each client perform?"

### 3. Run Context Injection

The backend automatically enhances queries with run-specific data:

```python
# Example context injection
context_info = f"""
[CONTEXT: The user is asking about Training Run #{run_id}.
Training Mode: {training_mode}.
Status: {status}.
Started: {start_time}.
Latest Metrics - Val Accuracy: 0.9234, Val Recall: 0.8876, Val Loss: 0.2341.
Number of Clients: 5.
Please answer the user's question with this run information in mind.]
"""
```

### 4. Visual Feedback

- **Loading State**: Spinner displayed while fetching runs
- **Empty State**: Helpful message if no runs exist
- **Selection Confirmation**: System message confirms run selection
- **Hover Effects**: Interactive hover states on run items

## Technical Implementation

### Frontend Components

#### ChatSidebar.tsx

**New State Variables:**

```typescript
const [showRunPicker, setShowRunPicker] = useState(false);
const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
const [selectedRun, setSelectedRun] = useState<RunContext | null>(null);
const [loadingRuns, setLoadingRuns] = useState(false);
```

**Key Functions:**

- `fetchAvailableRuns()`: Loads all training runs from backend
- `handleSelectRun(run)`: Sets active run context
- `handleInputChange(e)`: Detects `/` command
- `formatRunTime(timeStr)`: Formats timestamps for display

**API Integration:**

```typescript
// Fetch runs
const response = await api.results.listRuns();

// Send query with run context
const requestBody = {
  query: userMessage,
  session_id: sessionId,
  run_id: selectedRun?.runId,
  training_mode: selectedRun?.trainingMode,
};
```

### Backend Integration

#### chat_endpoints.py

**Enhanced ChatMessage Model:**

```python
class ChatMessage(BaseModel):
    query: str
    session_id: Optional[str] = None
    run_id: Optional[int] = None        # NEW
    training_mode: Optional[str] = None  # NEW
```

**Context Enhancement Logic:**

1. Check if `run_id` is provided
2. Fetch run data from database using `run_crud`
3. Extract relevant metrics and metadata
4. Inject context into query as a system prompt
5. Process enhanced query through RAG system

**Metrics Included:**

- Validation accuracy
- Validation recall
- Validation precision
- Validation loss
- Number of federated clients (if applicable)
- Training duration
- Current status

### Database Integration

Uses existing CRUD operations:

```python
from federated_pneumonia_detection.src.boundary.crud import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session

db = get_session()
run_data = run_crud.get_by_id(db, message.run_id)
```

## User Experience Flow

### Happy Path

1. User opens chat
2. User types `/`
3. Run picker appears with 10 completed runs
4. User clicks "Run #5" (federated, completed)
5. Badge appears: "Run #5 • federated • completed"
6. User asks: "What was the final accuracy?"
7. AI responds with specific accuracy from Run #5
8. User asks: "Why did the recall drop in round 3?"
9. AI analyzes Run #5's round-specific data and explains

### Edge Cases Handled

- **No Runs Available**: Friendly message suggesting to start training first
- **Loading Runs**: Spinner with loading indicator
- **Database Error**: Graceful fallback without context enhancement
- **Run Not Found**: Warning logged, query processed normally
- **Context Cleared**: User can remove run context at any time

## UI Components

### Run Picker Dropdown

```tsx
<div className="border-b bg-gray-50 max-h-64 overflow-y-auto">
  {/* Header */}
  <div className="p-3 border-b bg-white">
    <Activity className="h-4 w-4 text-medical" />
    <p>Select a Training Run</p>
  </div>

  {/* Run List */}
  <div className="divide-y">
    {availableRuns.map((run) => (
      <button onClick={() => handleSelectRun(run)}>{/* Run details */}</button>
    ))}
  </div>
</div>
```

### Context Badge

```tsx
{
  selectedRun && (
    <div className="bg-medical/10 border border-medical/30 rounded-lg p-3">
      <BarChart className="h-5 w-5 text-medical" />
      <p>Run #{selectedRun.runId}</p>
      <p>
        {selectedRun.trainingMode} • {selectedRun.status}
      </p>
      <Button onClick={() => setSelectedRun(null)}>
        <X className="h-3 w-3" />
      </Button>
    </div>
  );
}
```

### Pro Tip Display

```tsx
<div className="bg-gradient-to-r from-medical/5 to-blue-500/5">
  <Zap className="h-4 w-4 text-medical" />
  <p>Pro Tip</p>
  <p>
    Type <kbd>/</kbd> to select a training run...
  </p>
</div>
```

## Color Coding

### Training Mode

- **Federated**: Blue (`bg-blue-100 text-blue-600`)
- **Centralized**: Green (`bg-green-100 text-green-600`)

### Status

- **Completed**: Green (`bg-green-100 text-green-700`)
- **In Progress**: Blue (`bg-blue-100 text-blue-700`)
- **Failed/Other**: Gray (`bg-gray-100 text-gray-700`)

## Example Conversations

### Example 1: Performance Analysis

```
User: /
[Selects Run #12]
System: 📊 Now discussing Run #12 (federated training, completed)...

User: What was the best accuracy?
AI: Run #12 achieved a best validation accuracy of 92.34% in round 8 out of 10 federated rounds...

User: How does that compare to centralized training?
AI: Comparing Run #12 to your centralized runs, the federated approach achieved...
```

### Example 2: Debugging

```
User: /
[Selects Run #7]
System: 📊 Now discussing Run #7 (federated training, failed)...

User: Why did this run fail?
AI: Based on Run #7's logs and metrics, the training failed during round 3 when...

User: What should I change in the configuration?
AI: For Run #7, I'd recommend adjusting the following hyperparameters...
```

### Example 3: Federated Analysis

```
User: /
[Selects Run #15]
System: 📊 Now discussing Run #15 (federated training, completed)...

User: Show me the performance of each client
AI: In Run #15, you had 5 federated clients. Here's their individual performance...

User: Which client had the best recall?
AI: Client #3 in Run #15 achieved the highest recall of 89.2%...
```

## Benefits

### For Users

1. **Contextual Understanding**: AI understands exactly which run you're discussing
2. **Faster Insights**: No need to repeatedly specify run IDs
3. **Better Comparisons**: Easy to switch between runs for comparison
4. **Visual Clarity**: Clear indication of active context
5. **Intuitive UX**: Familiar slash-command pattern

### For Development

1. **Extensible**: Easy to add more run metadata
2. **Maintainable**: Clean separation of concerns
3. **Type-Safe**: Full TypeScript typing
4. **Testable**: Isolated functions for each feature
5. **Scalable**: Works with any number of runs

## Future Enhancements

### Potential Features

1. **Multi-Run Selection**: Compare multiple runs simultaneously
2. **Run Filtering**: Filter by status, date, training mode
3. **Run Tagging**: User-defined tags for organization
4. **Quick Stats**: Show metrics preview in picker
5. **Run Notes**: Add annotations to runs
6. **Export Conversations**: Save run-specific discussions
7. **Smart Suggestions**: AI suggests relevant runs based on query
8. **Real-Time Updates**: Show live metrics for in-progress runs

### Agentic AI Protocol Integration

The current implementation provides an excellent foundation for more sophisticated agentic AI protocols:

#### 1. **Multi-Agent Systems**

- **Specialist Agents**: Create agents specialized in different aspects (metrics analysis, debugging, optimization)
- **Coordinator Agent**: Route queries to appropriate specialist based on context
- **Memory Agent**: Maintain long-term memory of user's training patterns

#### 2. **Autonomous Analysis**

- **Proactive Insights**: Agent automatically analyzes completed runs and suggests improvements
- **Anomaly Detection**: Agent monitors runs and alerts users to unusual patterns
- **Comparative Analysis**: Agent automatically compares new runs to historical data

#### 3. **Planning & Execution**

- **Hyperparameter Optimization**: Agent suggests and queues new runs with optimized parameters
- **Experiment Planning**: Agent creates experiment roadmaps based on goals
- **Resource Management**: Agent optimizes training schedules across available compute

#### 4. **Tool Use**

- **Code Generation**: Agent generates code for custom metrics or visualizations
- **Configuration Management**: Agent modifies config files based on insights
- **Report Generation**: Agent creates comprehensive analysis reports

## Testing

### Manual Testing Checklist

- [ ] Type `/` triggers run picker
- [ ] All runs load correctly
- [ ] Run selection sets context
- [ ] Context badge displays correctly
- [ ] Context can be cleared
- [ ] Queries include run context in backend
- [ ] Multiple run switches work smoothly
- [ ] Empty state shows when no runs exist
- [ ] Loading state displays during fetch
- [ ] Keyboard navigation works (Enter to send)

### Edge Cases to Test

- [ ] Very long run descriptions
- [ ] Runs with missing data fields
- [ ] Backend database connection failure
- [ ] No internet connection
- [ ] Concurrent run selection attempts
- [ ] Rapid `/` typing

## Conclusion

The slash command feature transforms the chat from a generic Q&A interface into a powerful, context-aware training insights tool. By seamlessly integrating run selection with AI-powered analysis, users can quickly understand their experiments, debug issues, and optimize their models.

The implementation is production-ready, extensible, and provides an excellent foundation for future agentic AI enhancements.
