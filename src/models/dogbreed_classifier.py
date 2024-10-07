import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import timm

class DogBreedClassifier(L.LightningModule):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, list):
            x = torch.stack(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, list):
            x = torch.stack(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, list):
            x = torch.stack(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)