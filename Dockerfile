# Multi-stage. The final image carries the interpreter, the installed
# dependencies and this repository's source -- and no compiler, no pip cache,
# no build headers, no package index metadata.
#
# The reason is blast radius rather than image size. A build toolchain in a
# running container is what turns a code-execution foothold into a comfortable
# one: with gcc and pip present an attacker compiles and installs; without them
# they are working with what is already on disk.

# ---- Stage 1: build the dependency tree ------------------------------------
FROM python:3.12-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Present only so a wheel that has no manylinux build for this platform can be
# compiled here. It stays in this stage and never reaches the final image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-api.txt .

# A virtualenv rather than --user or a site-packages copy: one directory to
# move to the next stage, and the same layout in both.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements-api.txt

# pip, setuptools and wheel are build tooling and are removed from the venv
# before it is copied forward. An image inspection found them still present on
# the first build; leaving them there hands an attacker with code execution a
# package installer, which is most of the way to persistence. Nothing at
# runtime installs anything.
RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true  && rm -rf /opt/venv/lib/python3.12/site-packages/pip*            /opt/venv/lib/python3.12/site-packages/setuptools*            /opt/venv/lib/python3.12/site-packages/wheel*            /opt/venv/lib/python3.12/site-packages/pkg_resources            /opt/venv/bin/pip*

# ---- Stage 2: the runtime image --------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# scikit-learn's OpenMP runtime. Installed without the compiler that produced
# the wheel, which is the whole point of the split.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
 && rm -rf /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 \
           /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
           /usr/local/lib/python3.12/site-packages/setuptools \
           /usr/local/lib/python3.12/site-packages/pkg_resources \
           /usr/local/lib/python3.12/site-packages/wheel \
           /usr/local/lib/python3.12/ensurepip

COPY --from=build /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PDM_DATABASE=/data/pdm.db \
    PDM_RUN_STORE=/data/runs

WORKDIR /app
# .dockerignore is what keeps .env, data/raw, *.db and models/ out of this COPY.
# `tests/test_deployment.py` builds the image and inspects it rather than
# trusting that file.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser evals/prompts/ ./evals/prompts/
# Demo mode replays these. 4.8 MB of recorded runs plus the scenario
# definitions, which is what lets the page work in a container with no
# credential, no provider and no network. Without them `/` renders with no
# preset buttons -- the page would start, and be useless.
COPY --chown=appuser:appuser evals/transcripts/ ./evals/transcripts/
COPY --chown=appuser:appuser evals/scenarios.yaml ./evals/scenarios.yaml

# The database and the model artefacts are mounted at runtime, never baked in:
# the data is 77 MB of licensed download and the artefacts are rebuildable.
RUN mkdir -p /data/runs && chown -R appuser:appuser /data

USER appuser
EXPOSE 8000

# No shell form: exec form means the process is PID 1 and receives SIGTERM
# directly, so a container stop is a clean shutdown rather than a 10-second
# wait and a kill.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"]

ENTRYPOINT ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
