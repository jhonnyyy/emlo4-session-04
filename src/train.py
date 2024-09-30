import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.dogbreed import DogBreedDataModule
from models.dogbreed_classifier import DogBreedClassifier
import yaml

def main():
    # Load configuration
    with open("/app/config/config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Initialize DataModule
    data_module = DogBreedDataModule()

    # Initialize Model
    model = DogBreedClassifier()

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="/app/models",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # Initialize logger
    logger = TensorBoardLogger("/app/logs", name="dog_breed_classification")

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
    )

    # Train the model
    trainer.fit(model, data_module)

    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()