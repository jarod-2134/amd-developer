# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY agent.py ./

# Create input and output directories so local runs don't fail immediately 
# (In the evaluation environment, these will be mounted by the harness)
RUN mkdir -p /input /output

# Run the agent script when the container launches
CMD ["python", "agent.py"]
