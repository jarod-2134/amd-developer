#!/bin/bash
# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready (it usually takes a few seconds)
echo "Waiting for Ollama to start..."
sleep 5

# Run the python agent
echo "Running agent..."
python agent.py
