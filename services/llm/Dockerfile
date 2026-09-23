FROM medical-base:latest

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ۱. کپی هر دو فایل مورد نیاز به داخل کانتینر
COPY wheels/libgomp1.deb /tmp/libgomp1.deb
COPY wheels/gcc-12-base.deb /tmp/gcc-12-base.deb

# ۲. نصب پیش‌نیازها به صورت آفلاین
# ابتدا base نصب می‌شود و سپس libgomp1
RUN dpkg -i /tmp/gcc-12-base.deb && \
    dpkg -i /tmp/libgomp1.deb && \
    rm /tmp/gcc-12-base.deb /tmp/libgomp1.deb

# ۳. نصب پکیج‌های پایتون
COPY services/llm/requirements.txt /app/requirements.txt
RUN --mount=type=bind,source=wheels,target=/app/wheels \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r /app/requirements.txt

# ۴. کپی کدها
COPY services/llm/app /app/app

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
