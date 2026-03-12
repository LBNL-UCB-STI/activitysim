FROM python:3.10-slim

ENV ASIM_PATH=/activitysim
ENV EXAMPLE=prototype_mtc_clean
ENV EXEC_NAME=simulation.py
ENV PYTHONNOUSERSITE=1
ENV UV_NO_DEV=1
ENV PATH="$ASIM_PATH/.venv/bin:$PATH"
ENV PYTHONPATH="$ASIM_PATH:$PYTHONPATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR $ASIM_PATH

COPY pyproject.toml uv.lock README.md $ASIM_PATH/
RUN uv sync --locked --no-install-project --no-editable

COPY activitysim $ASIM_PATH/activitysim

RUN uv sync --locked --no-editable \
    && uv pip install DFO-LS geopandas

WORKDIR $ASIM_PATH/activitysim/examples/$EXAMPLE

ENTRYPOINT ["python", "-u", "simulation.py"]
