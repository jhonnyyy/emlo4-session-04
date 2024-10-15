# Build stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Copy the rest of the application
COPY . .

# Final stage
FROM python:3.12-slim-bookworm AS final

WORKDIR /app

# Create a non-root user
RUN useradd -m appuser

# Copy the installed packages and the application from the builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Make sure all scripts in src are executable
RUN chmod +x /app/src/*.py

# Add the current directory to PYTHONPATH
ENV PYTHONPATH="/app:$PYTHONPATH"

# Ensure Python output is sent straight to the container log
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Set the entrypoint to python
ENTRYPOINT ["python"]

# Set a default command (can be overridden)
CMD ["/app/src/train.py"]
