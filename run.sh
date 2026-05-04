#!/bin/bash

# Ensure the script stops if any command fails
set -e

echo "------------------------------------------------"
echo "Initializing RL2026HMMStock Project"
echo "------------------------------------------------"

# 1. Environment Setup
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating environment..."
source venv/bin/activate

# 2. Dependency Check
if [ -f "requirements.txt" ]; then
    echo "Updating dependencies..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    echo "Warning: requirements.txt not found. Skipping install."
fi

# 3. Execution Logic
# Based on Screenshot 2026-05-04 at 11.42.20 PM.jpg, 
# we'll run the discrete model first.
echo "Launching Discrete PPO HMM Model..."
python PPO_HMM_discrete.py

# Uncomment the following if you want to run the regime detector separately
# echo "Running Regime Detector..."
# python regime_detector.py

echo "------------------------------------------------"
echo "Process Finished Successfully"
echo "------------------------------------------------"