FROM python:3.10-slim

ENV ASIM_PATH=/activitysim
ENV EXAMPLE=prototype_mtc_clean
ENV EXEC_NAME=simulation.py
ENV PYTHONNOUSERSITE=1
ENV UV_NO_DEV=1
ENV PATH="$ASIM_PATH/.venv/bin:$PATH"
ENV PYTHONPATH="$ASIM_PATH${PYTHONPATH:+:$PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR $ASIM_PATH

# This Dockerfile assumes the build context is the parent directory that
# contains sibling `activitysim/` and `sharrow/` directories. Example:
#   docker build -f activitysim/Dockerfile -t zaneedell/activitysim:TAG .
COPY activitysim/pyproject.toml activitysim/uv.lock activitysim/README.md $ASIM_PATH/
COPY sharrow /tmp/sharrow-src
RUN uv sync --locked --no-install-project --no-editable \
    && uv pip install --no-deps /tmp/sharrow-src

COPY activitysim/activitysim $ASIM_PATH/activitysim

RUN uv sync --locked --no-editable \
    && uv pip install --no-deps /tmp/sharrow-src \
    && uv pip install DFO-LS geopandas zarr

WORKDIR $ASIM_PATH/activitysim/examples/$EXAMPLE

ENTRYPOINT ["python", "-u", "simulation.py"]
