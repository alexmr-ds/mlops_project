# Multi-stage build: install dependencies in a builder stage so the final
# image does not need uv or build tools at runtime.
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast, reproducible dependency resolution
RUN pip install --no-cache-dir uv==0.7.12

COPY pyproject.toml uv.lock ./
# Install only production dependencies (no dev extras) into the system Python
RUN uv sync --no-dev --no-editable --system

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy project source and the persisted Random Forest model bundle
COPY src/ ./src/
COPY data/06_models/random_forest_model.pkl ./data/06_models/random_forest_model.pkl

# Make the src package importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "mlops_project.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
