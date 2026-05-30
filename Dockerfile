FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/diffcfd

COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY diffcfd/ diffcfd/
COPY scripts/ scripts/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir maturin && \
    . "$HOME/.cargo/env" && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy scipy "gymnasium>=0.26" && \
    maturin develop --release

COPY Makefile .
COPY tests/ tests/

ENV PYTHONHASHSEED=42

CMD ["make", "flagship-b-ci"]
