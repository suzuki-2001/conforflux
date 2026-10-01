FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

# Which backbone to install: boltz | protenix | openfold3
ARG BACKBONE=boltz

WORKDIR /opt/conforflux

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY conforflux/    /opt/conforflux/conforflux/
COPY assets/        /opt/conforflux/assets/
COPY boltz/         /opt/conforflux/boltz/
COPY protenix/      /opt/conforflux/protenix/
COPY openfold3/     /opt/conforflux/openfold3/
COPY pyproject.toml README.md LICENSE /opt/conforflux/

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN set -eu; \
    case "${BACKBONE}" in \
      boltz) \
        pip install --no-cache-dir -e ./boltz; \
        pip install --no-cache-dir \
            "cuequivariance-ops-torch-cu12==0.9.1" \
            "cuequivariance-torch==0.9.1" \
            "cuequivariance-ops-cu12==0.9.1"; \
        pip install --no-cache-dir -e . ;; \
      protenix)   pip install --no-cache-dir -e ./protenix && pip install --no-cache-dir -e . ;; \
      openfold3)  pip install --no-cache-dir -e ./openfold3 && pip install --no-cache-dir -e . ;; \
      *) echo "BACKBONE must be boltz, protenix or openfold3 (got '${BACKBONE}')" >&2; exit 1 ;; \
    esac

ENV HOME=/root \
    NUMBA_CACHE_DIR=/tmp
ENV CONFORFLUX_BACKBONE=${BACKBONE}

# Dispatch to whichever CLI the selected backbone provides.
RUN printf '%s\n' '#!/bin/sh' \
    'case "$CONFORFLUX_BACKBONE" in' \
    '  boltz)     exec conforflux "$@" ;;' \
    '  protenix)  exec protenix "$@" ;;' \
    '  openfold3) exec run_openfold "$@" ;;' \
    'esac' > /usr/local/bin/entry && chmod +x /usr/local/bin/entry

ENTRYPOINT ["/usr/local/bin/entry"]
CMD ["--help"]
