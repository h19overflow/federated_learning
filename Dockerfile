# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv venv $VIRTUAL_ENV

# Copy the pre-generated requirements-all.txt
COPY requirements-all.txt ./

# Install all dependencies with uv
RUN uv pip install -r requirements-all.txt

# Install binary dependencies if you have them
COPY requirements-binary.txt* ./
RUN if [ -f requirements-binary.txt ]; then \
    pip install --no-cache-dir -r requirements-binary.txt; \
    fi

# Copy application code
COPY . .

# Final runtime stage
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy virtual environment and application
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose application port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
