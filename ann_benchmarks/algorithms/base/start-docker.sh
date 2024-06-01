#!/bin/bash

# Start Docker daemon in the background
dockerd &

# Wait for Docker daemon to start
while (! docker info > /dev/null 2>&1); do
    echo "Waiting for Docker to start..."
    sleep 1
done

# Execute the main process specified as CMD in the Dockerfile
python3 -u run_algorithm.py "$@"
