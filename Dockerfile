# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Add --no-cache-dir to reduce image size
# For GPU, you might need a different base image (e.g., nvidia/cuda) and specific torch install.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file into the container at /app
COPY main.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, if your app uses it)
ENV MODEL_NAME="Qodo/Qodo-Embed-1-1.5B"
ENV DEVICE="cpu"

# Run main.py when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]