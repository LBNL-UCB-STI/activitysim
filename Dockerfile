# syntax=docker/dockerfile:1.7

FROM python:3.10-slim AS builder

ENV ASIM_PATH=/activitysim \
    PYTHONNOUSERSITE=1 \
    UV_NO_DEV=1 \
    PATH="/activitysim/.venv/bin:$PATH" \
    PYTHONPATH="/activitysim${PYTHONPATH:+:$PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR $ASIM_PATH

# Build from the ActivitySim repo root and pass the local sharrow checkout as a
# named build context:
#   docker buildx build \
#     --build-context sharrow=../sharrow \
#     -f Dockerfile \
#     --platform linux/amd64 \
#     -t zaneedell/activitysim:TAG \
#     .
COPY pyproject.toml uv.lock README.md $ASIM_PATH/
COPY activitysim $ASIM_PATH/activitysim
COPY --from=sharrow . /tmp/sharrow-src

RUN uv sync --locked --no-editable \
    && uv pip install --no-deps /tmp/sharrow-src \
    && uv pip install --no-cache-dir DFO-LS geopandas zarr \
    && rm -rf /root/.cache /tmp/sharrow-src

FROM python:3.10-slim AS runtime

ENV ASIM_PATH=/activitysim \
    PYTHONNOUSERSITE=1 \
    UV_NO_DEV=1 \
    PATH="/activitysim/.venv/bin:$PATH" \
    PYTHONPATH="/activitysim${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR $ASIM_PATH

COPY --from=builder $ASIM_PATH/.venv $ASIM_PATH/.venv
COPY --from=builder $ASIM_PATH/activitysim $ASIM_PATH/activitysim

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "activitysim"]

# RUN COMMAND
#docker buildx build \
#  --platform linux/amd64 \
#  --build-context sharrow=../sharrow \
#  -f Dockerfile \
#  -t activitysim:local-amd64 \
#  --load .
