#!/bin/bash

# Print the current directory
echo "Current directory: $(pwd)"

# Check if virtual environment exists, create if it doesn't
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi
# Set API URL environment variable
echo "Setting API_URL environment variable..."
export API_URL="http://localhost:8002"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if not already installed
pip install -r requirements.txt

# Run the Streamlit application
echo "Starting Streamlit application..."
python -m streamlit run app.py

# Note: This file is only used for faster development purposes and should be done using Docker normal use.
