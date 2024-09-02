ARG PYTHON_VERSION=3.11
ARG FUNCTION_DIR="/home/app"
ARG UPSTREAM_URL="http://127.0.0.1:8080"
ARG WATCHDOG_MODE="http"

FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.10.5 as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:${PYTHON_VERSION}-slim-buster

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}

COPY --from=builder /usr/local /usr/local
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

COPY handler.py .
COPY classifier_int8.onnx .

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/lib/python* -name '__pycache__' -type d -exec rm -rf {} + || true

ENV fprocess="python3 handler.py" \
    cgi_headers="true" \
    mode=${WATCHDOG_MODE} \
    upstream_url=${UPSTREAM_URL} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    exec_timeout="60s" \
    write_timeout="15s" \
    read_timeout="15s"

HEALTHCHECK --interval=5s --timeout=10s --retries=3 CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]