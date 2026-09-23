FROM medical-base:latest

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY services/ingest/requirements.txt /app/requirements.txt

RUN --mount=type=bind,source=wheels,target=/app/wheels \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r /app/requirements.txt

COPY services/data/app /app/app

COPY services/ingest/ /app/

CMD ["python", "ingest.py"]
