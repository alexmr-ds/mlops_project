# Multi-stage build: install dependencies in a builder stage so the final
# image does not need uv or build tools at runtime.
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast, reproducible dependency resolution
RUN pip install --no-cache-dir uv==0.7.12

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# Install only production dependencies (no dev extras) into a project-local
# .venv, exactly as the locked uv.lock resolved them. --no-editable builds
# and installs mlops_project itself as a wheel, which needs README.md and
# src/ present (hatchling reads the README and packages src/mlops_project).
RUN uv sync --no-dev --no-editable --frozen

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

# uv sync always installs into .venv (there is no system-site-packages mode),
# so copy the whole virtual environment rather than system site-packages.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# mlops_project is already installed (non-editable) into .venv by the builder
# stage, so only the persisted Random Forest model bundle is needed here.
COPY data/06_models/random_forest_model.pkl ./data/06_models/random_forest_model.pkl

EXPOSE 8000

CMD ["uvicorn", "mlops_project.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
