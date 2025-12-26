# X-Ray Vision AI Forge 🖥️

**A modern, real-time dashboard for the Federated Pneumonia Detection System.**

This React application serves as the primary user interface for controlling training sessions, visualizing results, and monitoring system status.

---

## 🏗️ Architecture & Features

### Tech Stack
- **Framework**: [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Shadcn UI
- **State Management**: React Query + Context API
- **Visualization**: Recharts
- **Communication**: WebSocket (Real-time updates)

### Key Capabilities
- **Training Orchestration**: Start centralized or federated runs via simple forms.
- **Live Monitoring**: Watch training metrics (Loss, Accuracy, F1) stream in real-time.
- **Results Analysis**: View confusion matrices, ROC curves, and detailed per-epoch stats.
- **Chat Assistant**: Interact with the Arxiv Agent for research questions.

---

## 📂 Directory Structure

```
src/
├── components/          # Reusable UI components
│   ├── dashboard/       # Main dashboard widgets
│   ├── training/        # Training control forms
│   └── ui/              # Shadcn primitive components
├── context/             # React Context (Auth, Theme)
├── hooks/               # Custom hooks
│   └── useWebSocket.ts  # Real-time metrics connection
├── lib/                 # Utilities and helpers
├── pages/               # Route page components
├── services/            # API client services
│   └── api.ts           # Axios configuration
└── types/               # TypeScript definitions
```

---

## 🔌 WebSocket Integration

The dashboard connects to the backend WebSocket server (`ws://localhost:8765`) to receive live updates without polling.

**Hook Location**: `src/hooks/use-websocket.tsx`

**Message Handling**:
- `training_mode`: Sets the UI to "Centralized" or "Federated" mode.
- `epoch_end`: Updates charts with new epoch data points.
- `round_end`: Updates federated learning progress bars.
- `training_end`: Triggers final results fetch.

---

## 🚀 Getting Started

### Prerequisites
- Node.js 20+
- npm or bun

### Installation

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Access Dashboard**
   Open [http://localhost:8080](http://localhost:8080)

### Configuration
Update `.env` to point to your backend API:
```env
VITE_API_URL=http://localhost:8001
VITE_WS_URL=ws://localhost:8765
```

---

## 🤝 Backend Connection

This frontend requires the Python backend to be running:
1. **API Server**: `http://localhost:8001`
2. **WebSocket Server**: `ws://localhost:8765`

Ensure both services are active before starting a training session.