# Use a smaller base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt /app/
COPY src /app/src/

# Install dependencies and clean up in one layer
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kaggle && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    mkdir -p /app/data /app/models /app/logs && \
    chmod +x /app/src/run_all.sh && \
    apt-get purge -y --auto-remove gcc libc6-dev

# Set the entrypoint to the run_all.sh script
ENTRYPOINT ["/app/src/run_all.sh"]