from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import wandb

wandb.init(project='UNet')
wandb.run.name = 'UNet_no_aug_8down'
wandb.run.save()

class UnetModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging SSIM and PSNR across batches
        self.train_ssim = MeanMetric()
        self.val_ssim = MeanMetric()
        self.test_ssim = MeanMetric()

        self.train_psnr = MeanMetric()
        self.val_psnr = MeanMetric()
        self.test_psnr = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation SSIM
        self.val_ssim_best = MaxMetric()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_ssim.reset()
        self.val_ssim_best.reset()
        self.val_psnr.reset()


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        restored = self.forward(x)
        loss = self.criterion(restored, y)
        return loss, restored, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, restored, targets = self.model_step(batch)

        # Calculate SSIM and PSNR
        ssim_value = structural_similarity_index_measure(restored, targets)
        psnr_value = peak_signal_noise_ratio(restored, targets)

        # update and log metrics
        self.train_loss(loss)
        self.train_ssim(ssim_value)
        self.train_psnr(psnr_value)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ssim", self.train_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/psnr", self.train_psnr, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({
            "train/loss": self.train_loss.compute(), 
            "train/ssim": self.train_ssim.compute(), 
            "train/psnr": self.train_psnr.compute()
        })

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, restored, targets = self.model_step(batch)

        # Calculate SSIM and PSNR
        ssim_value = structural_similarity_index_measure(restored, targets)
        psnr_value = peak_signal_noise_ratio(restored, targets)

        # update and log metrics
        self.val_loss(loss)
        self.val_ssim(ssim_value)
        self.val_psnr(psnr_value)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({
            "val/loss": self.val_loss.compute(), 
            "val/ssim": self.val_ssim.compute(), 
            "val/psnr": self.val_psnr.compute()
        })

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        ssim = self.val_ssim.compute()  # get current val SSIM
        self.val_ssim_best(ssim)  # update best so far val SSIM
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/ssim_best", self.val_ssim_best.compute(), sync_dist=True, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, restored, targets = self.model_step(batch)

        # Calculate SSIM and PSNR
        ssim_value = structural_similarity_index_measure(restored, targets)
        psnr_value = peak_signal_noise_ratio(restored, targets)

        # update and log metrics
        self.test_loss(loss)
        self.test_ssim(ssim_value)
        self.test_psnr(psnr_value)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", self.test_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({
            "test/loss": self.test_loss.compute(), 
            "test/ssim": self.test_ssim.compute(), 
            "test/psnr": self.test_psnr.compute()
        })

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = UnetModule(None, None, None, None)
