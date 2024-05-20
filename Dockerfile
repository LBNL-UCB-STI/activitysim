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

RUN git clone -b telework https://github.com/LBNL-UCB-STI/activitysim.git

RUN conda install -n base conda-libmamba-solver

RUN conda env create --quiet -p $FULL_CONDA_PATH --file activitysim/environment.yml --solver=libmamba

RUN cd activitysim && git pull && $FULL_CONDA_PATH/bin/python setup.py install

ENV PATH $FULL_CONDA_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV

ENV EXAMPLE bay_area

WORKDIR $ASIM_PATH/$EXAMPLE

RUN git pull

ENTRYPOINT ["python", "-u", "simulation.py"]