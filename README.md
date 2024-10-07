# Dog Breed Classification

This project uses a deep learning model to classify dog breeds.

## Prerequisites

- Docker installed on your system
- Sufficient disk space (at least 10GB recommended)

## How to Use Docker

### Build the Docker Image

```bash
docker build -t dogbreed .
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
   docker build --no-cache -t dogbreed .
   ```

### Run the Entire Pipeline

This command will run the training, evaluation, and inference in sequence:

1. Training

docker run -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs dogbreed src/train.py

2. Evaluation 

docker run -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs dogbreed src/eval.py

3. Inference 

docker run -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs dogbreed src/infer.py

## Volume Mounts Explained

- `/app/data`: Directory to store the dataset and test images (maps to `./data` in the project)
- `/app/models`: Directory to store trained models (maps to `./src/models` in the project)
- `/app/logs`: Directory to store TensorBoard logs (maps to `./logs` in the project)

Make sure these directories exist in your project structure before running the Docker command.

## Configuration

IN the config module, there are all the config files for the project. 

Add the kaggle username and apikey to @configs/data/dog_breed.yaml file.

## For testing using coverage 

# to run the test
coverage run -m pytest

# to check the coverage report
coverage report -m

For more detailed instructions, refer to the Docker documentation.
