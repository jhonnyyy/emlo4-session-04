import pytest
import torch
import rootutils
import os
from hydra import initialize_config_dir, compose

# Setup the root of the project
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
from src.models.dogbreed_classifier import DogBreedClassifier

@pytest.fixture(scope="module")
def cfg():
    config_dir = os.path.join(root, "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train")
    return cfg

def test_dogbreed_classifier(cfg):
    num_classes = cfg.data.extra.num_classes
    model = DogBreedClassifier(
        num_classes=num_classes,
        learning_rate=cfg.data.extra.learning_rate,
        model_name=cfg.model.get("name", "resnet50"),
        pretrained=cfg.model.get("pretrained", True)
    )
    
    # Test model initialization
    assert isinstance(model, DogBreedClassifier)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, num_classes)

def test_dogbreed_classifier_training_step(cfg):
    num_classes = cfg.data.extra.num_classes
    model = DogBreedClassifier(
        num_classes=num_classes,
        learning_rate=cfg.data.extra.learning_rate,
        model_name=cfg.model.get("name", "resnet50"),
        pretrained=cfg.model.get("pretrained", True)
    )
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = model.training_step((input_tensor, labels), 0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss.requires_grad