FROM registry.access.redhat.com/ubi9/python-311

USER 0

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PIP_NO_CACHE_DIR=off 
ENV PIP_DISABLE_PIP_VERSION_CHECK=on 
ENV PIP_DEFAULT_TIMEOUT=100 
# ENV POETRY_VERSION=1.0.3 
ENV POETRY_HOME="/opt/poetry" 
ENV POETRY_VIRTUALENVS_IN_PROJECT=false 
ENV POETRY_NO_INTERACTION=1 
ENV PYSETUP_PATH="/tmp/pysetup" 
ENV VENV_PATH="/tmp/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN poetry --version

USER 1001

WORKDIR /opt/app-root/src

RUN git clone https://github.com/noelo/svcs-rag

WORKDIR /opt/app-root/src/svcs-rag

# RUN poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR
