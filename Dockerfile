# Start from the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables:
# PYTHONUNBUFFERED=1: Ensures Python output is sent straight to terminal (useful for Docker logs)
# PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files to disk
# NVIDIA_* variables ensure the NVIDIA Container Toolkit injects GPU drivers properly
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/usr/local/bin:${PATH}"

# Install basic system utilities needed for building and fetching
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install `uv` (the fast Python package installer and resolver)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the pyproject.toml and uv.lock files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Optional: W&B environment configuration placeholders (can be overridden by .env or --env)
ENV WANDB_API_KEY=""

# Defer dependency installation to runtime to keep the Docker image small.
# 'uv run python' will automatically download dependencies when executed.

# Copy the entire remaining project directory from the host into the container's /app directory
COPY . .

