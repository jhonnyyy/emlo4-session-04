# Use an official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir kaggle

# Copy the entire src directory into the container
COPY src /app/src/

# Copy the configuration file
COPY src/datamodules/config.yaml /app/config/config.yaml

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Set the entrypoint to python
ENTRYPOINT ["python"]

# Set the default command to run the training script
CMD ["/app/src/train.py"]