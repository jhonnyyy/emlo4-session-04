import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import timm
import yaml

class DogBreedClassifier(L.LightningModule):
    def __init__(self, config_path: str = "src/datamodules/config.yaml"):
        super().__init__()
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        self.num_classes = self.config['model']['num_classes']
        self.learning_rate = float(self.config['training']['learning_rate'])  # Convert to float
        
        self.save_hyperparameters()
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=self.num_classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)