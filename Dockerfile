FROM continuumio/miniconda3

ENV CONDA_DIR /opt/conda
ENV CONDA_ENV asim
ENV FULL_CONDA_PATH $CONDA_DIR/envs/$CONDA_ENV


ENV ASIM_PATH /activitysim
ENV ASIM_SUBDIR examples
ENV EXEC_NAME simulation.py

RUN apt-get --allow-releaseinfo-change update \
	&& apt-get install -y build-essential zip unzip
RUN conda update conda --yes
RUN conda install -n base conda-libmamba-solver --yes
RUN conda config --set solver libmamba

RUN git clone -b merge-latest-asim https://github.com/LBNL-UCB-STI/activitysim.git

RUN conda env create --quiet -p $FULL_CONDA_PATH --file activitysim/conda-environments/activitysim-dev.yml --solver=libmamba
RUN cd activitysim && $FULL_CONDA_PATH/bin/python setup.py install

ENV PATH $FULL_CONDA_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV

ENV EXAMPLE example_mtc

WORKDIR $ASIM_PATH/$ASIM_PATH/$ASIM_SUBDIR/$EXAMPLE
ENTRYPOINT ["python", "-u", "simulation.py"]