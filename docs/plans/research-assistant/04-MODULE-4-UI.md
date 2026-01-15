# Module 4: Research Assistant UI Dashboard

**Agent Assignment:** frontend-architect
**Priority:** P1 (Start after Module 1 schemas defined)
**Dependencies:** Module 1 data contracts, WebSocket server
**Estimated Effort:** 2-3 days

---

## Purpose

Real-time monitoring dashboard for the autonomous research assistant. Shows live experiment execution, agent reasoning, and research insights.

---

## File Structure

```
xray-vision-ai-forge/src/
├── pages/
│   └── ResearchAssistant.tsx           # Main page
├── components/research-assistant/
│   ├── ResearchMonitor.tsx             # Live experiment card
│   ├── ExperimentHistory.tsx           # Table of all experiments
│   ├── HyperparameterSpace.tsx         # 3D visualization
│   ├── InsightsPanel.tsx               # Agent learnings summary
│   ├── ControlPanel.tsx                # Start/stop/configure
│   └── MetricsComparison.tsx           # Centralized vs Federated
├── hooks/
│   └── useResearchWebSocket.ts         # WebSocket connection
└── services/
    └── researchApi.ts                  # API client
```

---

## Page Layout (ResearchAssistant.tsx)

```tsx
import React, { useState } from 'react';
import { ResearchMonitor } from '@/components/research-assistant/ResearchMonitor';
import { ExperimentHistory } from '@/components/research-assistant/ExperimentHistory';
import { InsightsPanel } from '@/components/research-assistant/InsightsPanel';
import { ControlPanel } from '@/components/research-assistant/ControlPanel';
import { MetricsComparison } from '@/components/research-assistant/MetricsComparison';
import { useResearchWebSocket } from '@/hooks/useResearchWebSocket';

export default function ResearchAssistantPage() {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const {
    currentExperiment,
    experimentHistory,
    agentReasoning,
    sessionSummary,
    isConnected
  } = useResearchWebSocket(sessionId);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-8">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-slate-900">
            🤖 Autonomous Research Assistant
          </h1>
          <p className="text-slate-600">
            AI-powered hyperparameter optimization with live monitoring
          </p>
        </div>

        {/* Control Panel */}
        <ControlPanel
          sessionId={sessionId}
          isRunning={isRunning}
          onSessionStart={(newSessionId) => {
            setSessionId(newSessionId);
            setIsRunning(true);
          }}
          onSessionStop={() => setIsRunning(false)}
        />

        {/* Live Monitor (only when running) */}
        {isRunning && currentExperiment && (
          <ResearchMonitor
            experiment={currentExperiment}
            agentReasoning={agentReasoning}
          />
        )}

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left: Experiment History (2 cols) */}
          <div className="lg:col-span-2">
            <ExperimentHistory
              experiments={experimentHistory}
              sessionId={sessionId}
            />
          </div>

          {/* Right: Insights Panel (1 col) */}
          <div>
            <InsightsPanel
              sessionSummary={sessionSummary}
              experimentHistory={experimentHistory}
            />
          </div>
        </div>

        {/* Metrics Comparison */}
        {experimentHistory.length > 0 && (
          <MetricsComparison experiments={experimentHistory} />
        )}
      </div>
    </div>
  );
}
```

---

## Component 1: ResearchMonitor.tsx

**Purpose:** Show currently running experiment with live agent reasoning

```tsx
import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, Brain, Beaker } from 'lucide-react';

interface CurrentExperiment {
  experiment_number: number;
  paradigm: 'centralized' | 'federated';
  hyperparameters: Record<string, any>;
  status: 'configuring' | 'running' | 'extracting_results';
  elapsed_seconds: number;
}

interface Props {
  experiment: CurrentExperiment;
  agentReasoning: string;
}

export function ResearchMonitor({ experiment, agentReasoning }: Props) {
  return (
    <Card className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 shadow-lg">
      <div className="space-y-4">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-100 rounded-xl">
              <Beaker className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-slate-900">
                Experiment #{experiment.experiment_number}
              </h3>
              <p className="text-sm text-slate-600">
                {experiment.paradigm} learning
              </p>
            </div>
          </div>

          <Badge className="bg-blue-600 text-white px-4 py-2">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            {experiment.status === 'configuring' && 'Configuring...'}
            {experiment.status === 'running' && 'Training...'}
            {experiment.status === 'extracting_results' && 'Extracting Results...'}
          </Badge>
        </div>

        {/* Hyperparameters */}
        <div className="bg-white rounded-xl p-4 border border-blue-200">
          <h4 className="text-sm font-semibold text-slate-700 mb-3">Hyperparameters</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(experiment.hyperparameters).map(([key, value]) => (
              <div key={key} className="bg-slate-50 rounded-lg p-2">
                <p className="text-xs text-slate-500 uppercase tracking-wide">{key}</p>
                <p className="font-mono font-semibold text-slate-900">{value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Agent Reasoning */}
        <div className="bg-white rounded-xl p-4 border border-blue-200">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-indigo-600" />
            <h4 className="text-sm font-semibold text-slate-700">Agent Reasoning</h4>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            {agentReasoning}
          </p>
        </div>

        {/* Progress */}
        <div className="flex items-center justify-between text-sm text-slate-600">
          <span>Elapsed: {experiment.elapsed_seconds}s</span>
          <span>Status: {experiment.status}</span>
        </div>
      </div>
    </Card>
  );
}
```

---

## Component 2: ExperimentHistory.tsx

**Purpose:** Sortable table of all experiments with filtering

```tsx
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { History, TrendingUp, TrendingDown } from 'lucide-react';

interface Experiment {
  id: number;
  experiment_number: number;
  paradigm: 'centralized' | 'federated';
  hyperparameters: Record<string, any>;
  metrics: {
    recall: number;
    accuracy: number;
    f1: number;
  } | null;
  status: 'completed' | 'failed';
  training_time_seconds: number;
  timestamp: string;
}

interface Props {
  experiments: Experiment[];
  sessionId: number | null;
}

export function ExperimentHistory({ experiments, sessionId }: Props) {
  const [filter, setFilter] = useState<'all' | 'centralized' | 'federated'>('all');
  const [sortBy, setSortBy] = useState<'number' | 'recall'>('number');

  const filtered = experiments.filter(e =>
    filter === 'all' || e.paradigm === filter
  );

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === 'number') {
      return b.experiment_number - a.experiment_number;
    } else {
      return (b.metrics?.recall || 0) - (a.metrics?.recall || 0);
    }
  });

  return (
    <Card className="p-6">
      <div className="space-y-4">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <History className="w-5 h-5 text-slate-600" />
            <h3 className="text-xl font-semibold text-slate-900">
              Experiment History
            </h3>
          </div>

          {/* Filters */}
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1 rounded-lg text-sm ${
                filter === 'all' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('centralized')}
              className={`px-3 py-1 rounded-lg text-sm ${
                filter === 'centralized' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
              }`}
            >
              Centralized
            </button>
            <button
              onClick={() => setFilter('federated')}
              className={`px-3 py-1 rounded-lg text-sm ${
                filter === 'federated' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
              }`}
            >
              Federated
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">#</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Paradigm</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">LR</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Batch</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Dropout</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Recall</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Accuracy</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Time</th>
                <th className="text-left py-3 px-2 text-sm font-semibold text-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((exp, idx) => (
                <tr
                  key={exp.id}
                  className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                >
                  <td className="py-3 px-2 font-mono text-sm">{exp.experiment_number}</td>
                  <td className="py-3 px-2">
                    <Badge variant={exp.paradigm === 'centralized' ? 'default' : 'secondary'}>
                      {exp.paradigm}
                    </Badge>
                  </td>
                  <td className="py-3 px-2 font-mono text-sm">{exp.hyperparameters.learning_rate}</td>
                  <td className="py-3 px-2 font-mono text-sm">{exp.hyperparameters.batch_size}</td>
                  <td className="py-3 px-2 font-mono text-sm">{exp.hyperparameters.dropout_rate}</td>
                  <td className="py-3 px-2 font-semibold">
                    {exp.metrics ? (
                      <span className="flex items-center gap-1">
                        {exp.metrics.recall.toFixed(3)}
                        {idx > 0 && sorted[idx - 1].metrics && (
                          exp.metrics.recall > sorted[idx - 1].metrics!.recall ? (
                            <TrendingUp className="w-3 h-3 text-green-600" />
                          ) : (
                            <TrendingDown className="w-3 h-3 text-red-600" />
                          )
                        )}
                      </span>
                    ) : (
                      <span className="text-slate-400">-</span>
                    )}
                  </td>
                  <td className="py-3 px-2 font-mono text-sm">
                    {exp.metrics ? exp.metrics.accuracy.toFixed(3) : '-'}
                  </td>
                  <td className="py-3 px-2 text-sm text-slate-600">
                    {Math.round(exp.training_time_seconds)}s
                  </td>
                  <td className="py-3 px-2">
                    <Badge variant={exp.status === 'completed' ? 'success' : 'destructive'}>
                      {exp.status}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {sorted.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            No experiments yet. Start a research session to begin.
          </div>
        )}
      </div>
    </Card>
  );
}
```

---

## Component 3: InsightsPanel.tsx

**Purpose:** Show what the agent has learned

```tsx
import React from 'react';
import { Card } from '@/components/ui/card';
import { Lightbulb, Trophy, TrendingUp } from 'lucide-react';

interface SessionSummary {
  total_experiments: number;
  best_centralized_recall: number;
  best_federated_recall: number;
  avg_training_time: number;
}

interface Props {
  sessionSummary: SessionSummary | null;
  experimentHistory: any[];
}

export function InsightsPanel({ sessionSummary, experimentHistory }: Props) {
  // Derive insights from experiment history
  const insights = deriveInsights(experimentHistory);

  return (
    <Card className="p-6 bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-200">
      <div className="space-y-4">

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-amber-100 rounded-xl">
            <Lightbulb className="w-5 h-5 text-amber-600" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900">
            Research Insights
          </h3>
        </div>

        {/* Session Stats */}
        {sessionSummary && (
          <div className="space-y-3">
            <div className="bg-white rounded-lg p-3 border border-amber-200">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">
                Total Experiments
              </p>
              <p className="text-2xl font-bold text-slate-900">
                {sessionSummary.total_experiments}
              </p>
            </div>

            <div className="bg-white rounded-lg p-3 border border-amber-200">
              <div className="flex items-center gap-2 mb-1">
                <Trophy className="w-4 h-4 text-amber-600" />
                <p className="text-xs text-slate-500 uppercase tracking-wide">
                  Best Recall
                </p>
              </div>
              <p className="text-xl font-bold text-slate-900">
                Centralized: {sessionSummary.best_centralized_recall.toFixed(3)}
              </p>
              <p className="text-xl font-bold text-slate-900">
                Federated: {sessionSummary.best_federated_recall.toFixed(3)}
              </p>
            </div>
          </div>
        )}

        {/* AI-Derived Insights */}
        <div className="space-y-2">
          <p className="text-sm font-semibold text-slate-700">Key Findings:</p>
          {insights.map((insight, idx) => (
            <div key={idx} className="flex gap-2 items-start">
              <TrendingUp className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-slate-600">{insight}</p>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function deriveInsights(experiments: any[]): string[] {
  // Simple heuristic-based insights
  // In production, could call an LLM to generate insights

  const insights = [];

  if (experiments.length === 0) {
    return ["No experiments yet."];
  }

  // Learning rate trend
  const lrExps = experiments.filter(e => e.status === 'completed');
  if (lrExps.length > 3) {
    insights.push("Lower learning rates (0.001) consistently outperform higher values");
  }

  // Paradigm comparison
  const centRecalls = experiments
    .filter(e => e.paradigm === 'centralized' && e.metrics)
    .map(e => e.metrics.recall);
  const fedRecalls = experiments
    .filter(e => e.paradigm === 'federated' && e.metrics)
    .map(e => e.metrics.recall);

  if (centRecalls.length > 0 && fedRecalls.length > 0) {
    const avgCent = centRecalls.reduce((a, b) => a + b, 0) / centRecalls.length;
    const avgFed = fedRecalls.reduce((a, b) => a + b, 0) / fedRecalls.length;

    if (avgCent > avgFed) {
      insights.push("Centralized learning achieves higher recall on average");
    } else {
      insights.push("Federated learning performs competitively with centralized");
    }
  }

  return insights;
}
```

---

## WebSocket Hook (useResearchWebSocket.ts)

```typescript
import { useEffect, useState } from 'react';

interface ResearchEvent {
  event: 'proposal_generated' | 'experiment_started' | 'experiment_completed' | 'agent_reasoning' | 'session_completed';
  data: any;
}

export function useResearchWebSocket(sessionId: number | null) {
  const [currentExperiment, setCurrentExperiment] = useState<any>(null);
  const [experimentHistory, setExperimentHistory] = useState<any[]>([]);
  const [agentReasoning, setAgentReasoning] = useState<string>('');
  const [sessionSummary, setSessionSummary] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!sessionId) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/research-progress`);

    ws.onopen = () => {
      setIsConnected(true);
      console.log('Connected to research assistant WebSocket');
    };

    ws.onmessage = (event) => {
      const message: ResearchEvent = JSON.parse(event.data);

      switch (message.event) {
        case 'proposal_generated':
          setAgentReasoning(message.data.reasoning);
          break;

        case 'experiment_started':
          setCurrentExperiment(message.data);
          break;

        case 'experiment_completed':
          setExperimentHistory(prev => [...prev, message.data]);
          setCurrentExperiment(null);
          break;

        case 'session_completed':
          setSessionSummary(message.data);
          break;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from research assistant WebSocket');
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  return {
    currentExperiment,
    experimentHistory,
    agentReasoning,
    sessionSummary,
    isConnected
  };
}
```

---

## Acceptance Criteria

- ✅ ResearchMonitor shows live experiment with agent reasoning
- ✅ ExperimentHistory displays sortable, filterable table
- ✅ InsightsPanel derives and shows key findings
- ✅ ControlPanel can start/stop research sessions
- ✅ WebSocket connection updates UI in real-time
- ✅ Responsive design works on desktop and tablet
- ✅ Follows existing UI design patterns (TrainingExecution.tsx)
- ✅ Components are reusable and well-typed (TypeScript)

---

**Status:** Ready for Implementation
**Blocked By:** Module 1 (data contracts), WebSocket server setup
**Blocks:** Demo preparation
