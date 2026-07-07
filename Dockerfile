# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install curl and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Start Ollama in the background, pull the Qwen 2.5 7B model, and then kill Ollama so the layer saves
RUN ollama serve & sleep 5 && ollama pull qwen2.5:7b && pkill ollama

# Copy the rest of the application code
COPY agent.py ./
COPY start.sh ./
RUN chmod +x start.sh

# Create input and output directories so local runs don't fail immediately 
# (In the evaluation environment, these will be mounted by the harness)
RUN mkdir -p /input /output

# Run the startup script when the container launches
CMD ["./start.sh"]
