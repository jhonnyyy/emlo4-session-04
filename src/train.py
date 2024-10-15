# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from src.models.dogbreed_classifier import DogBreedClassifier
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import os
# import sys
# import rootutils
# from hydra.utils import instantiate

# root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

# @hydra.main(version_base="1.3", config_path="../configs", config_name="train")
# def main(cfg: DictConfig):

#     print(OmegaConf.to_yaml(cfg))
#     print(f"Current working directory: {os.getcwd()}")
#     print(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
#     print(f"Sys path: {sys.path}")

#     # Initialize DataModule
#     data_module = instantiate(cfg.data.datamodule)
#     print("Successfully instantiated data_module")

#     # Call prepare_data to download the dataset if necessary
#     data_module.prepare_data()
#     print("Prepared data successfully")
    
#     # Set up the data module
#     data_module.setup()

#     # Initialize Model
#     model_config = {
#         "model_name": cfg.model.model.name,
#         "pretrained": cfg.model.model.pretrained,
#         "num_classes": cfg.data.extra.num_classes,
#         "learning_rate": cfg.data.extra.learning_rate
#     }
#     model = DogBreedClassifier(**model_config)

#     # Initialize callbacks
#     callbacks = []
#     if "callbacks" in cfg:
#         for cb_name, cb_conf in cfg.callbacks.items():
#             if "_target_" in cb_conf:
#                 try:
#                     callback = hydra.utils.instantiate(cb_conf)
#                     callbacks.append(callback)
#                 except TypeError as e:
#                     print(f"Warning: Could not instantiate callback {cb_name}. Error: {e}")

#     # Initialize logger
#     logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name=cfg.task_name)

#     # Initialize Trainer
#     trainer_config = {k: v for k, v in cfg.trainer.items() if k != '_target_'}
#     trainer = L.Trainer(
#         callbacks=callbacks,
#         logger=logger,
#         **trainer_config
#     )

#     # Train the model
#     trainer.fit(model, data_module)

#     # Print the path of the best model checkpoint
#     if callbacks and isinstance(callbacks[0], ModelCheckpoint):
#         print(f"Best model checkpoint saved at: {callbacks[0].best_model_path}")

# if __name__ == "__main__":
#     main()


import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from src.models.dogbreed_classifier import DogBreedClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import rootutils

root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"Sys path: {sys.path}")

    # Initialize DataModule

    data_module = hydra.utils.instantiate(cfg.data.datamodule)
    print("Successfully instantiated data_module")

    # Call prepare_data to download the dataset if necessary
    data_module.prepare_data()
    print("Prepared data successfully")

    # Set up the data module
    data_module.setup()

    # Initialize Model
    model_config = {
        "model_name": cfg.model.model.name,
        "pretrained": cfg.model.model.pretrained,
        "num_classes": cfg.data.extra.num_classes,
        "learning_rate": cfg.data.extra.learning_rate
    }
    model = DogBreedClassifier(**model_config)

    # Initialize callbacks
    callbacks = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                try:
                    callback = hydra.utils.instantiate(cb_conf)
                    callbacks.append(callback)
                except TypeError as e:
                    print(f"Warning: Could not instantiate callback {cb_name}. Error: {e}")

    # Initialize logger
    logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name=cfg.task_name)

    # Initialize Trainer
    trainer_config = {k: v for k, v in cfg.trainer.items() if k != '_target_'}
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config
    )

    # Train the model
    trainer.fit(model, data_module)

    # Print the path of the best model checkpoint
    if callbacks and isinstance(callbacks[0], ModelCheckpoint):
        print(f"Best model checkpoint saved at: {callbacks[0].best_model_path}")

if __name__ == "__main__":
    main()
