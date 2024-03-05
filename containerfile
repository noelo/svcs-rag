FROM registry.access.redhat.com/ubi9/python-311

USER 1001

RUN pip install poetry

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PIP_NO_CACHE_DIR=off 
ENV PIP_DISABLE_PIP_VERSION_CHECK=on 
ENV PIP_DEFAULT_TIMEOUT=100 
ENV POETRY_HOME="/opt/poetry" 
ENV POETRY_VIRTUALENVS_CREATE=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true 
ENV POETRY_NO_INTERACTION=1 
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

RUN mkdir -p /opt/app-root/data

COPY chromadb /opt/app-root/data

WORKDIR /opt/app-root/src

RUN git clone https://github.com/noelo/svcs-rag

WORKDIR /opt/app-root/src/svcs-rag

RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

COPY --chmod=755 start-chainlit.sh /opt/app-root/src/svcs-rag/start-chainlit.sh

ENV CHROMA_DB_LOCN=/opt/app-root/data

ENTRYPOINT /opt/app-root/src/svcs-rag/start-chainlit.sh
