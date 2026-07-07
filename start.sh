#!/bin/bash
# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready before running the agent
echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is ready (after ${i}s)."
        break
    fi
    sleep 1
done

# Run the python agent
echo "Running agent..."
python agent.py
