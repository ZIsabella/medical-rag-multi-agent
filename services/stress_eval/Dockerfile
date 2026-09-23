FROM medical-base:latest

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# کپی فایل نیازمندی‌ها
COPY services/stress_eval/requirements.txt /app/requirements.txt

# نصب پکیج‌ها از فولدر wheels (که در ریشه پروژه قرار دارد)
RUN --mount=type=bind,source=wheels,target=/app/wheels \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r /app/requirements.txt

# کپی اسکریپت استرس
COPY services/stress_eval/stress.py /app/stress.py

CMD ["python", "stress.py"]
