import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from src.models.dogbreed_classifier import DogBreedClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import rootutils

root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory: {os.getcwd()}")

    # Initialize DataModule
    data_module = hydra.utils.instantiate(cfg.data.datamodule)
    data_module.setup(stage="test")

    # Load the model from checkpoint
    checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
    
    # Sort checkpoint files by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # Use the most recent checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)

    # Initialize logger
    logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name="evaluation")

    # Initialize Trainer
    trainer_config = {k: v for k, v in cfg.trainer.items() if k != '_target_'}
    trainer = L.Trainer(
        logger=logger,
        **trainer_config
    )

    # Evaluate the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()