ARG PYTHON_VERSION=3.11
ARG FUNCTION_DIR="/home/app"
ARG UPSTREAM_URL="http://127.0.0.1:8080"
ARG WATCHDOG_MODE="http"

FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.10.5 as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:${PYTHON_VERSION}-slim-buster

WORKDIR ${FUNCTION_DIR}

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

COPY handler.py .
COPY classifier_int8.onnx .

ENV fprocess="python3 handler.py" \
    cgi_headers="true" \
    mode=${WATCHDOG_MODE} \
    upstream_url=${UPSTREAM_URL} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    exec_timeout="900s" 

HEALTHCHECK --interval=5s --timeout=10s --retries=3 CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]