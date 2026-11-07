# Backdrop server (web app + import worker) on fe-pro; see deploy/fe-pro/docker-compose.yml and fe-pro's SERVER.md.
# Code and Python's files stay root-owned; the containers run as backdrop (2008) and write only the data directory.
# uid 2008 has no /etc/passwd entry in the image, and torch looks up the user name on import, so USER/LOGNAME are set.
FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app HOME=/tmp HF_HUB_OFFLINE=1 \
    USER=backdrop LOGNAME=backdrop TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor TRITON_CACHE_DIR=/tmp/triton
WORKDIR /app
COPY deploy/fe-pro/requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
COPY pipeline ./pipeline
COPY worker ./worker
COPY webapp ./webapp
# config.json is per machine (gitignored); the compose file mounts fe-pro's read-only at /app/config.json.
CMD ["python3", "webapp/app.py"]
