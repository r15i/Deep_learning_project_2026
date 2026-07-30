#!/bin/bash
exec > test_jobs.log 2>&1

echo "=== Testing Local Jobs ==="
echo "Running: make job-all EPOCHS=1"
make job-all EPOCHS=1
echo "Job-all exit code: $?"

echo "=== Testing Docker Jobs ==="
echo "Running: make docker-build"
make docker-build
echo "Docker-build exit code: $?"

echo "Running: make train-local EPOCHS=1"
make train-local EPOCHS=1
echo "Train-local exit code: $?"

echo "Running: make test-local"
make test-local
echo "Test-local exit code: $?"

echo "=== Testing Remote Jobs ==="
echo "Running: make all-hyperstack EPOCHS=1"
make all-hyperstack EPOCHS=1
echo "All-hyperstack exit code: $?"

echo "Done!"
