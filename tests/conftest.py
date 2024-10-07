import pytest
import os
import rootutils
from hydra import initialize_config_dir, compose

# Setup the root of the project
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

@pytest.fixture(scope="session")
def cfg():
    config_dir = os.path.join(root, "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train")
    # Add default values for model name and pretrained
    if "model" not in cfg:
        cfg.model = {}
    if "name" not in cfg.model:
        cfg.model.name = "resnet50"
    if "pretrained" not in cfg.model:
        cfg.model.pretrained = True
    return cfg

@pytest.fixture(scope="session")
def mock_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    for class_name in ["Golden Retriever", "German Shepherd", "Labrador Retriever"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(5):  # Create 5 images per class
            image_path = class_dir / f"image{i}.jpg"
            create_dummy_image(image_path)
    return data_dir

def create_dummy_image(file_path):
    import numpy as np
    from PIL import Image
    array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(file_path)
