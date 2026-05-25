FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

WORKDIR /opt/conforflux

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY boltz/         /opt/conforflux/boltz/
COPY conforflux/    /opt/conforflux/conforflux/
COPY examples/      /opt/conforflux/examples/
COPY assets/        /opt/conforflux/assets/
COPY pyproject.toml README.md LICENSE /opt/conforflux/

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir -e ./boltz \
 && pip install --no-cache-dir \
        "cuequivariance-ops-torch-cu12==0.9.1" \
        "cuequivariance-torch==0.9.1" \
        "cuequivariance-ops-cu12==0.9.1" \
 && pip install --no-cache-dir -e .

ENV HOME=/root \
    NUMBA_CACHE_DIR=/tmp

ENTRYPOINT ["conforflux"]
CMD ["--help"]
