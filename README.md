# Dog Breed Classification

This project uses a deep learning model to classify dog breeds.

## Prerequisites

- Docker installed on your system
- Kaggle API credentials (kaggle.json file)

## How to Use Docker

### Build the Docker Image

```bash
docker build -t dogbreed-classifier .
```

### Train the Model

```bash
docker run -v /path/to/your/kaggle.json:/root/.kaggle/kaggle.json:ro \
           -v /path/to/save/data:/app/data \
           -v /path/to/save/models:/app/models \
           dogbreed-classifier /app/src/train.py
```

### Evaluate the Model

```bash
docker run -v /path/to/your/data:/app/data:ro \
           -v /path/to/your/models:/app/models:ro \
           dogbreed-classifier /app/src/eval.py
```

### Run Inference

First, make sure you have test images in the `/path/to/your/test/images` directory.

```bash
docker run -v /path/to/your/models:/app/models:ro \
           -v /path/to/your/test/images:/app/data/test_images:ro \
           dogbreed-classifier /app/src/infer.py
```

Note: Replace `/path/to/your/...` with the actual paths on your system.

## Volume Mounts Explained

- `/root/.kaggle/kaggle.json`: Your Kaggle API credentials (read-only)
- `/app/data`: Directory to store the dataset
- `/app/models`: Directory to store trained models
- `/app/data/test_images`: Directory containing test images for inference

Make sure these directories exist on your host machine before running the Docker commands.