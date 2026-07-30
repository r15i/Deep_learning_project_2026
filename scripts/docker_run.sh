#!/bin/bash
# Run docker container

DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/r15i/nndl-project:latest"}

docker rm -f nndl_worker 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected. Running with --gpus all"
    docker run -it --rm --name nndl_worker --gpus all \
        --hostname local --network host --env-file .env \
        -v $(pwd)/dataset:/app/dataset \
        -v $(pwd)/weights:/app/weights \
        ${DOCKER_IMAGE} /app/scripts/bootstrap.sh "$@"
else
    echo "No GPU detected. Running on CPU"
    docker run -it --rm --name nndl_worker \
        --hostname local --network host --env-file .env \
        -v $(pwd)/dataset:/app/dataset \
        -v $(pwd)/weights:/app/weights \
        ${DOCKER_IMAGE} /app/scripts/bootstrap.sh "$@"
fi
