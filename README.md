# Dog Breed Classification

This project uses a deep learning model to classify dog breeds.

## Prerequisites

- Docker installed on your system
- Sufficient disk space (at least 10GB recommended)

## How to Use Docker

### Build the Docker Image

```bash
docker build -t dogbreed-classifier .
```

If you encounter disk space issues, try the following:

1. Clean up Docker resources:
   ```
   docker system prune -a
   docker volume prune
   ```

2. Clear Docker build cache:
   ```
   docker builder prune -a
   ```

3. Build without cache:
   ```
   docker build --no-cache -t dogbreed-classifier .
   ```

### Run the Entire Pipeline

This command will run the training, evaluation, and inference in sequence:

```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/src/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           dogbreed-classifier
```

## Volume Mounts Explained

- `/app/data`: Directory to store the dataset and test images (maps to `./data` in the project)
- `/app/models`: Directory to store trained models (maps to `./src/models` in the project)
- `/app/logs`: Directory to store TensorBoard logs (maps to `./logs` in the project)

Make sure these directories exist in your project structure before running the Docker command.

## Configuration

The `config.yaml` file is located at `src/datamodules/config.yaml`. Ensure this file is properly configured before building the Docker image.

## Troubleshooting

If you're experiencing disk space issues:

1. Check available disk space: Use `df -h` to view available space.
2. Increase Docker's disk space allocation in Docker Desktop settings.
3. Use a `.dockerignore` file to exclude unnecessary files from the build context.

For more detailed instructions, refer to the Docker documentation.

## Additional Notes

- Ensure that your `requirements.txt` file includes only necessary dependencies.
- Place your test images in the `./data/test_images` directory before running the Docker container.