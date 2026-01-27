# Docker Setup

## Quick Start

```bash
cd docker
cp .env.example .env
# Edit .env with your values
docker compose up -d
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL with pgvector |
| backend | 8001 | FastAPI server |
| frontend | 5173 | React app (nginx) |
| federated | - | Flower simulation (optional) |

## Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f backend

# Run federated learning
docker compose --profile federated up federated

# Rebuild after code changes
docker compose build --no-cache backend

# Stop all
docker compose down

# Stop and remove volumes
docker compose down -v
```

## GPU Support

Requires NVIDIA Container Toolkit. Remove `deploy.resources.reservations` from docker-compose.yml for CPU-only mode.
