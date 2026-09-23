FROM medical-base:latest

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY services/orchestrator/requirements.txt /app/requirements.txt

RUN --mount=type=bind,source=wheels,target=/app/wheels \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r /app/requirements.txt

COPY services/orchestrator/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
