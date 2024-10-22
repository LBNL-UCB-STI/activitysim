FROM --platform=linux/amd64 continuumio/miniconda3

ENV CONDA_DIR /opt/conda
ENV CONDA_ENV asim
ENV FULL_CONDA_PATH $CONDA_DIR/envs/$CONDA_ENV


ENV ASIM_PATH /activitysim
ENV ASIM_SUBDIR examples
ENV EXEC_NAME simulation.py

RUN apt-get --allow-releaseinfo-change update \
	&& apt-get install -y build-essential zip unzip
RUN conda update conda --yes

RUN echo "UPDATE again again"

RUN wget https://raw.githubusercontent.com/LBNL-UCB-STI/activitysim/3cdd7a8d622f0636af3f601105a92a5c0d978420/environment.yml

RUN conda install -n base conda-libmamba-solver

RUN conda env create --quiet -p $FULL_CONDA_PATH --file environment.yml --solver=libmamba

RUN apt-get upgrade git -y

RUN git config --global http.postBuffer 157286400

RUN export GIT_TRACE_PACKET=1
RUN export GIT_TRACE=1
RUN export GIT_CURL_VERBOSE=1
RUN git config --global core.compression 0

RUN echo "Reset 31"

RUN git clone --depth 1 -b beam-plans-fixes https://github.com/LBNL-UCB-STI/activitysim.git



RUN cd activitysim && git pull && $FULL_CONDA_PATH/bin/python setup.py install


ENV PATH $FULL_CONDA_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV

ENV EXAMPLE bay_area

WORKDIR $ASIM_PATH/$EXAMPLE

RUN echo "Update"



ENTRYPOINT ["python", "-u", "simulation.py"]