FROM --platform=linux/amd64 continuumio/miniconda3 as builder

ENV CONDA_DIR /opt/conda
ENV CONDA_ENV asim
ENV FULL_CONDA_PATH $CONDA_DIR/envs/$CONDA_ENV


ENV ASIM_PATH /activitysim
ENV ASIM_SUBDIR examples
ENV EXEC_NAME simulation.py

# Install system dependencies and configure build flags
RUN apt-get --allow-releaseinfo-change update \
    && apt-get install -y build-essential zip unzip gcc g++ wget\
    && rm -rf /var/lib/apt/lists/* \
    && export MAKEFLAGS="-j$(nproc)"
# Update conda and configure pip
RUN conda update conda --yes \
    && conda install -n base conda-libmamba-solver \
    && pip config set global.no-cache-dir true

RUN conda install -n base conda-libmamba-solver

RUN mkdir -p $ASIM_PATH/conda-environments && \
    wget -O $ASIM_PATH/conda-environments/activitysim-dev.yml https://raw.githubusercontent.com/LBNL-UCB-STI/activitysim/refs/heads/new-merge-lbl/conda-environments/activitysim-dev.yml
RUN sed -i '/-e \.\./d' $ASIM_PATH/conda-environments/activitysim-dev.yml \
    && conda env create -p $FULL_CONDA_PATH --file $ASIM_PATH/conda-environments/activitysim-dev.yml --solver=libmamba \
    && $FULL_CONDA_PATH/bin/pip install --only-binary pandas "pandas>=1.4.0,<2"

RUN conda install -c conda-forge dfo-ls --solver=libmamba
ENV EXAMPLE prototype_mtc_clean

COPY activitysim/abm $ASIM_PATH/activitysim/abm
COPY activitysim/core $ASIM_PATH/activitysim/core
COPY activitysim/cli $ASIM_PATH/activitysim/cli
COPY activitysim/examples/example_manifest.yaml $ASIM_PATH/activitysim/examples/example_manifest.yaml
COPY activitysim/examples/__init__.py $ASIM_PATH/activitysim/examples/__init__.py
COPY activitysim/examples/$EXAMPLE $ASIM_PATH/activitysim/examples/$EXAMPLE
COPY activitysim/__init__.py $ASIM_PATH/activitysim/
COPY pyproject.toml $ASIM_PATH/pyproject.toml

RUN $FULL_CONDA_PATH/bin/pip install --no-deps --only-binary :all: $ASIM_PATH/

ENV PATH $FULL_CONDA_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV
ENV PYTHONPATH $ASIM_PATH:$PYTHONPATH



WORKDIR $ASIM_PATH/activitysim/examples/$EXAMPLE

ENTRYPOINT ["python", "-u", "simulation.py"]