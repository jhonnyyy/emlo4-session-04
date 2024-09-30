import lightning as L
from datamodules.dogbreed import DogBreedDataModule
from models.dogbreed_classifier import DogBreedClassifier
import glob
import os

def main():
    # Initialize DataModule
    data_module = DogBreedDataModule()
    data_module.setup()

    # Find the latest checkpoint
    checkpoints = glob.glob("/app/models/best-checkpoint*.ckpt")
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in /app/models/")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

    # Load the best model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(latest_checkpoint)

    # Initialize Trainer
    trainer = L.Trainer(accelerator="auto")

    # Run evaluation
    results = trainer.test(model, datamodule=data_module)

    # Print validation metrics
    print("Evaluation Results:")
    for k, v in results[0].items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()