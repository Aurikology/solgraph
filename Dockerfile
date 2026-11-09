# SolGraph API + graph page. `docker compose up -d --build`, then open http://127.0.0.1:8000
FROM python:3.13.0-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# the kit goes in without its pins, as in the README (they would hold LangChain at 0.3)
RUN pip install --no-deps solana-agent-kit-py==1.4.0
COPY pyproject.toml README.md ./
COPY solgraph ./solgraph
RUN pip install ".[redis]"

# no keys are baked in: the AI key comes from .env at run time, and SolGraph never needs a wallet key
RUN useradd --create-home solgraph
USER solgraph
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "solgraph.api:app", "--host", "0.0.0.0", "--port", "8000"]
