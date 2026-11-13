FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_OFFLINE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# LightGBM utilise la bibliothèque OpenMP, absente de l'image Python slim.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

RUN useradd --create-home --uid 10001 scorer
COPY --chown=scorer:scorer app/ ./app/
USER scorer

EXPOSE 7860
CMD ["python", "-m", "app.main"]
