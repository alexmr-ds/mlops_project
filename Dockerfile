# Multi-stage build: install dependencies in a builder stage so the final
# image does not need uv or build tools at runtime.
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast, reproducible dependency resolution
RUN pip install --no-cache-dir uv==0.7.12

COPY pyproject.toml uv.lock ./
# Install only production dependencies (no dev extras) into a project-local
# .venv, exactly as the locked uv.lock resolved them.
RUN uv sync --no-dev --no-editable --frozen

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

# uv sync always installs into .venv (there is no system-site-packages mode),
# so copy the whole virtual environment rather than system site-packages.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy project source and the persisted Random Forest model bundle
COPY src/ ./src/
COPY data/06_models/random_forest_model.pkl ./data/06_models/random_forest_model.pkl

# Make the src package importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "mlops_project.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
