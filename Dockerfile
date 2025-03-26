FROM --platform=linux/amd64 continuumio/miniconda3 as builder

ENV CONDA_DIR /opt/conda
ENV CONDA_ENV asim
ENV FULL_CONDA_PATH $CONDA_DIR/envs/$CONDA_ENV


ENV ASIM_PATH /activitysim
ENV ASIM_SUBDIR examples
ENV EXEC_NAME simulation.py

# Install system dependencies and configure build flags
RUN apt-get --allow-releaseinfo-change update \
    && apt-get install -y build-essential zip unzip gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && export MAKEFLAGS="-j$(nproc)"
# Update conda and configure pip
RUN conda update conda --yes \
    && conda install -n base conda-libmamba-solver \
    && pip config set global.no-cache-dir true

RUN conda install -n base conda-libmamba-solver

COPY activitysim $ASIM_PATH/activitysim
COPY conda-environments/ $ASIM_PATH/conda-environments/
RUN sed -i '/-e \.\./d' $ASIM_PATH/conda-environments/activitysim-dev.yml \
    && conda env create -p $FULL_CONDA_PATH --file $ASIM_PATH/conda-environments/activitysim-dev.yml --solver=libmamba \
    && $FULL_CONDA_PATH/bin/pip install --only-binary pandas "pandas>=1.4.0,<2"

COPY pyproject.toml $ASIM_PATH/pyproject.toml

RUN $FULL_CONDA_PATH/bin/pip install --no-deps --only-binary :all: $ASIM_PATH/

ENV PATH $FULL_CONDA_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV
ENV PYTHONPATH $ASIM_PATH:$PYTHONPATH

ENV EXAMPLE prototype_mtc

WORKDIR $ASIM_PATH/activitysim/examples/$EXAMPLE

ENTRYPOINT ["python", "-u", "simulation.py"]