import pytest
import sys
import torch
import lightning as L
from hydra.utils import instantiate
import rootutils
from hydra.core.hydra_config import HydraConfig
from unittest.mock import patch, MagicMock
from src.train import main
from omegaconf import DictConfig, OmegaConf
from src.models.dogbreed_classifier import DogBreedClassifier

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
sys.path.append(root)

def test_model_instantiation(cfg):
    model_config = {
        "num_classes": cfg.data.extra.num_classes,
        "learning_rate": cfg.data.extra.learning_rate,
        "model_name": cfg.model.get("name", "resnet50"),
        "pretrained": cfg.model.get("pretrained", True)
    }
    model = DogBreedClassifier(**model_config)
    assert isinstance(model, DogBreedClassifier)

def test_datamodule_instantiation(cfg, mock_data_dir):
    cfg.data.datamodule.data_dir = str(mock_data_dir)
    data_module = instantiate(cfg.data.datamodule)
    assert data_module is not None

def test_trainer_instantiation(cfg):
    with patch.object(HydraConfig, 'get', return_value=cfg):
        trainer_config = {k: v for k, v in cfg.trainer.items() if k != '_target_'}
        trainer = L.Trainer(**trainer_config)
        assert isinstance(trainer, L.Trainer)

def test_model_forward_pass(cfg):
    model_config = {
        "num_classes": cfg.data.extra.num_classes,
        "learning_rate": cfg.data.extra.learning_rate,
        "model_name": cfg.model.get("name", "resnet50"),
        "pretrained": cfg.model.get("pretrained", True)
    }
    model = DogBreedClassifier(**model_config)
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, cfg.data.extra.num_classes)

def test_config_consistency(cfg):
    assert "data" in cfg
    assert "model" in cfg
    assert "trainer" in cfg
    assert "paths" in cfg
    assert "callbacks" in cfg
    assert "logger" in cfg

    assert cfg.task_name == "dog_breed_classification"
    assert cfg.ignore_warnings is True

@patch('src.train.hydra.utils.instantiate')
@patch('src.train.L.Trainer')
@patch('src.train.TensorBoardLogger')
@patch('src.train.DogBreedClassifier')
@patch('hydra.core.hydra_config.HydraConfig.get')
def test_main_function(mock_hydra_config, mock_model, mock_logger, mock_trainer, mock_instantiate, cfg):
    mock_hydra_config.return_value = cfg
    mock_data_module = MagicMock()
    mock_instantiate.return_value = mock_data_module
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance
    
    cfg.paths.log_dir = "dummy_log_dir"
    cfg.model.model = {"name": "resnet50", "pretrained": False}
    cfg.data.extra = {"num_classes": 10, "learning_rate": 0.001}
    cfg.trainer = {"max_epochs": 10}
    cfg.task_name = "test_task"
    
    main(cfg)
    
    mock_instantiate.assert_called()
    mock_data_module.prepare_data.assert_called_once()
    mock_data_module.setup.assert_called_once()
    mock_model.assert_called_once()
    mock_logger.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer_instance.fit.assert_called_once()

@patch('src.train.hydra.utils.instantiate')
@patch('src.train.L.Trainer')
def test_callback_initialization(mock_trainer, mock_instantiate, cfg):
    mock_instantiate.return_value = MagicMock()
    mock_trainer.return_value = MagicMock()
    
    cfg.callbacks = {
        "callback1": {"_target_": "lightning.pytorch.callbacks.EarlyStopping"},
        "callback2": {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"}
    }
    cfg.trainer = {"max_epochs": 10}
    
    main(cfg)
    
    assert mock_instantiate.call_count >= 2

@patch('src.train.DogBreedClassifier')
@patch('src.train.hydra.utils.instantiate')
@patch('src.train.L.Trainer')
def test_model_config_creation(mock_trainer, mock_instantiate, mock_model, cfg):
    mock_instantiate.return_value = MagicMock()
    mock_trainer.return_value = MagicMock()
    
    cfg.model.model = {"name": "resnet50", "pretrained": False}
    cfg.data.extra = {"num_classes": 10, "learning_rate": 0.001}
    cfg.trainer = {"max_epochs": 10}
    
    main(cfg)
    
    mock_model.assert_called_once_with(
        model_name="resnet50",
        pretrained=False,
        num_classes=10,
        learning_rate=0.001
    )

@patch('src.train.L.Trainer')
def test_trainer_config_creation(mock_trainer, cfg):
    cfg.trainer = {
        "_target_": "lightning.Trainer",
        "max_epochs": 10,
        "accelerator": "cpu"
    }
    cfg.callbacks = {}  # Empty callbacks to avoid instantiation errors
    
    main(cfg)
    
    mock_trainer.assert_called_once()
    _, kwargs = mock_trainer.call_args
    assert "max_epochs" in kwargs
    assert kwargs["max_epochs"] == 10
    assert "accelerator" in kwargs
    assert kwargs["accelerator"] == "cpu"