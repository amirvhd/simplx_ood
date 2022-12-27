import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional import auroc, accuracy
from models.new_model import WideResNet


class classifier_module(pl.LightningModule):

    def __init__(self, n_classes=10, lr=5e-1, wd=0, n_layers=0, drop=0.3, **kwargs):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = wd
        self.n_cls = n_classes
        self.num_freeze_layers = n_layers
        self.drop = drop
        self.results = None
        self.model = WideResNet(dropout_rate=self.drop)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images):
        # images = self.upsample(images)
        output = self.model(images)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": y}

    # add train step end
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        class_roc_auc = auroc(predictions, labels, num_classes=self.n_cls)
        acc = accuracy(probs, labels, num_classes=self.n_cls, top_k=1)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar(f"roc_auc/Train", class_roc_auc, self.current_epoch)
        self.logger.experiment.add_scalar(f"acc/Train", acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs.detach(), "labels": y}

    # add val epoch end
    def validation_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        acc = accuracy(probs, labels, num_classes=self.n_cls)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": y}

    def test_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        acc = accuracy(probs, labels, num_classes=self.n_cls)
        self.results = torch.max(probs, dim=-1).values.cpu().numpy()

        self.log("test_acc", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                    weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=1e-5)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def get_results(self):
        return self.results
