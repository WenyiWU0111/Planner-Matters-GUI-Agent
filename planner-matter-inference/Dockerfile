# Use NVIDIA CUDA base image with Python 3.10 (devel for building flash_attn)
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements files first (for better caching)
COPY requirements.txt .

# Install PyTorch first (required by other packages)
RUN pip install --no-cache-dir torch==2.7.1

# Install vllm first (it will pull in compatible transformers version)
RUN pip install --no-cache-dir vllm==0.10.0

# Build flash_attn from source against installed torch (slow but reliable)
ENV USE_NINJA=1
RUN python -m pip install --no-build-isolation flash_attn==2.3.3 

# Install other ML/AI packages
RUN pip install --no-cache-dir peft wandb qwen_vl_utils qwen_agent openai anthropic

# Install basic Python dependencies
RUN pip install --no-cache-dir beartype playwright Pillow pandas gymnasium Matplotlib dashscope bs4 \
    python-dotenv faiss-cpu evaluate nltk crawl4ai accelerate dotenv

# Install web/networking libraries
RUN pip install --no-cache-dir aiofiles aiohappyeyeballs aiohttp aiolimiter aiosignal aiosqlite \
    Brotli certifi chardet charset-normalizer httpcore httpx requests urllib3 websocket-client

# Install data processing libraries
RUN pip install --no-cache-dir datasets pyarrow protobuf

# Install other utilities
RUN pip install --no-cache-dir patchright tf-playwright-stealth beautifulsoup4 lxml cssselect \
    click click-log GitPython rich pydantic tiktoken tqdm humanize sentry-sdk


# Install Playwright and its dependencies
RUN playwright install --with-deps chromium

# Copy the rest of the project
COPY . .

# Set the default command
CMD ["/bin/bash"]
