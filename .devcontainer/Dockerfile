FROM python:3.12-slim

ENV PATH=/root/.local/bin:$PATH \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /workspace

# apt-get 
RUN apt-get update && \
    apt-get install -y sudo git curl zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
