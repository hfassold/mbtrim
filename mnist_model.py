from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from simple_dense_net import SimpleDenseNet

import mbtrim as mbtrim

# FAH This is a "pytorch lightning" module, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
# and https://github.com/PyTorchLightning/pytorch-lightning
class MNISTLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SimpleDenseNet(hparams=self.hparams)

        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any, is_training_step: bool = False):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, is_training_step = True)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        # FAH: Note that for a pytorch lightning module, 'training_step' is expected to return at least some values (The 'loss' value is mandatory !!).
        # FAH: See https://github.com/PyTorchLightning/pytorch-lightning/issues/1256 and https://blog.ceshine.net/post/pytorch-lightning-grad-accu/ for more information
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # 'outputs' is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "acc": acc,  "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # See https://forums.pytorchlightning.ai/t/understanding-logging-and-validation-step-validation-epoch-end/291/2
        # and https://stackoverflow.com/questions/67182475/what-is-the-difference-between-on-validation-epoch-end-and-validation-epoch-e
        # 'outputs' is a list of dicts returned from `validation_step`
        # Calculate
        xyz = 123

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # hard-code for now
        lr = 0.001
        weight_decay = 0.0005
        return torch.optim.Adam(
            params=self.parameters(), lr=lr, weight_decay=weight_decay
        )


# FAH The'boss'version of 'MNistLitModel', with following extensions:
# - Added 'minibatch-trimming' during training (trims a mini-batch by discarding the
#   samples in the mini-batch with the _lowest_ losses)
class MNISTLitModelBoss(MNISTLitModel):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        lr: float = 0.001,
        enable_mbt: bool = False,
        mbt_a: float = 1.0,
        mbt_b: float = 0.2,
        mbt_epoch_scheme: str = 'linear'
    ):
        super().__init__(input_size=input_size, lin1_size = lin1_size, lin2_size = lin2_size, output_size = output_size)
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # We have to add a variant of the loss metric which does NOT do a reduction (via 'mean') across the minibatch
        # Via setting reduction='none', the loss function skips the reduction (mean) over the batch
        # This parameter seems to be supported by all pytorch loss functions, see https://pytorch.org/docs/stable/nn.html#loss-functions
        self.criterion_no_reduce = torch.nn.CrossEntropyLoss(reduction='none')

    # I only have to override the 'step' function which calculates the loss
    # On the relation between 'loss.backward' and 'optimizer.step', see our WIKI at https://digital-wiki.joanneum.at/pages/viewpage.action?pageId=127501025
    # Note both operations are called automatically in pytorch lightning, see the training 'pseudo-code' at https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    def step(self, batch: Any, is_training_step: bool = False):
        # Parameter controlling whether mini-batch 'trimming' is enabled or not
        enable_mbt = self.hparams['enable_mbt']
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        if not is_training_step or not enable_mbt:
            # for test and validation step (or when minibatch trimming is disabled), proceed in the standard way
            loss = self.criterion(logits, y)
        else:
            # In training step with active 'mini-batch trimming', we have to proceed differently.
            # We use the variant of the loss metric which does NOT do a reduction (via 'mean') across the minibatch
            loss = self.criterion_no_reduce(logits, y)
            # Retrieve the loss for the 'r' samples with the _highest_ loss in the mini-batch,
            # the other samples (with lower loss)in the mini-batch are 'trimmed' away.
            # That is the core principle: We calculate the gradient only from the 'r' samples
            # in the mini-batch with the highest loss.
            # So we are focusing during training on the 'more difficult' samples ...
            loss = mbtrim.get_adapted_loss_for_trimmed_minibatch(loss, self.trainer.current_epoch, self.trainer.max_epochs,
                            self.hparams['mbt_a'], self.hparams['mbt_b'], self.hparams['mbt_epoch_scheme'])
            # Final step is to do a reduction (via 'torch.mean') over all r remaining samples.
            loss = torch.mean(loss)

        return loss, preds, y



