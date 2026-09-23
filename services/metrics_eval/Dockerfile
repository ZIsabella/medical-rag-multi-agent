FROM medical-base:latest

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY services/metrics_eval/requirements.txt /app/requirements.txt

RUN --mount=type=bind,source=wheels,target=/app/wheels \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r /app/requirements.txt

COPY services/metrics_eval/evaluate_rag.py /app/evaluate_rag.py

CMD ["python", "evaluate_rag.py"]
