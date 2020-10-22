FROM conda/miniconda3
WORKDIR usr/src/dqn
COPY . .
RUN conda env create -f env.yml
CMD ['source activate', 'env']