import pytest
import os
import hydra
from omegaconf import DictConfig
import rootutils
from unittest.mock import patch
from PIL import Image
import numpy as np
from hydra import initialize_config_dir, compose

# Setup the root of the project
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.datamodules.dogbreed import DogBreedDataModule

def create_dummy_image(file_path):
    # Create a small dummy image
    array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(file_path)

@pytest.fixture
def mock_image_folder(tmp_path):
    # Create a mock dataset structure with more images
    data_dir = tmp_path / "data"
    for class_name in ["class1", "class2", "class3"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(10):  # Create 10 images per class
            image_path = class_dir / f"image{i}.jpg"
            create_dummy_image(image_path)
    return data_dir

@pytest.fixture(scope="module")
def cfg():
    config_dir = os.path.join(root, "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train")
    return cfg

def test_dogbreed_datamodule(cfg, mock_image_folder):
    # Override batch_size and data_dir in the config for testing
    cfg.data.datamodule.batch_size = 4
    cfg.data.datamodule.data_dir = str(mock_image_folder)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    
    datamodule.prepare_data()
    datamodule.setup()
    
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    assert total_size == 30  # 3 classes * 10 images per class
    assert len(datamodule.train_dataset) >= len(datamodule.val_dataset)
    assert len(datamodule.train_dataset) >= len(datamodule.test_dataset)
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Check if we can iterate through the dataloader
    for batch in train_loader:
        assert len(batch) == 2  # (images, labels)
        assert batch[0].shape[0] == cfg.data.datamodule.batch_size
        break
