#!/bin/bash

# Print the current directory
echo "Current directory: $(pwd)"

# Check if virtual environment exists, create if it doesn't
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
# pip install -r requirements.txt

# Run the FastAPI application
echo "Starting FastAPI application on port 8002..."
uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload

# Note: This file is only used for faster development purposes and should be Dockerized normal use.
