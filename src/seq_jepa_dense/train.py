"""Pretrain SimCLR, seq-JEPA or seq-JEPA-dense on PathoROB Tolkach ESCA.

    python -m seq_jepa_dense.train --method seqjepa-dense

Checkpoints and metrics are written to <output-dir>/<method>/; the paper
evaluates the final checkpoint, last.ckpt.
"""
import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy

from seq_jepa_dense.common import METHODS, build_model
from seq_jepa_dense.data import TolkachESCA, make_loader

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--data-path", default="datasets/tolkach_esca")
    p.add_argument("--input-size", type=int, default=128, help="encoder input size in px")
    p.add_argument("--num-crops", type=int, default=5,
                   help="crops per sequence S = N + 1 for seq-JEPA (SimCLR always uses 2)")
    # SimCLR
    p.add_argument("--temperature", type=float, default=0.5)
    # seq-JEPA
    p.add_argument("--action-emb-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3, help="transformer aggregator layers")
    p.add_argument("--pred-hidden", type=int, default=1024, help="predictor hidden dimension")
    # optimization
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--warmup-start-lr", type=float, default=1e-5)
    p.add_argument("--precision", default="bf16-mixed")
    # misc
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--save-every", type=int, default=10, help="checkpoint frequency in epochs")
    p.add_argument("--resume", default=None, help="checkpoint to resume training from")
    args = p.parse_args()
    if args.method == "simclr":
        args.num_crops = 2
    return args


class OnlineProbe(pl.Callback):
    """Monitoring only: after every epoch, trains a linear classifier on the
    frozen encoder representation of single crops for one more epoch and
    reports its train and test accuracy. Not used for the reported results."""

    def __init__(self, train_loader, test_loader, num_classes, lr=1e-3):
        super().__init__()
        self.train_loader, self.test_loader = train_loader, test_loader
        self.num_classes, self.lr = num_classes, lr
        self.probe = None

    def _run_epoch(self, encoder, loader, device, train):
        accuracy = MulticlassAccuracy(self.num_classes, average="micro").to(device)
        for batch in loader:
            x, y = batch["crops"][:, 0].to(device), batch["label"].to(device)
            with torch.no_grad():
                z = encoder(x)
            with torch.set_grad_enabled(train):
                logits = self.probe(z)
            if train:
                self.optimizer.zero_grad()
                F.cross_entropy(logits, y).backward()
                self.optimizer.step()
            accuracy.update(logits.detach(), y)
        return 100.0 * accuracy.compute().item()

    def on_train_epoch_end(self, trainer, pl_module):
        encoder = pl_module.model.encoder
        if self.probe is None:
            self.probe = nn.Linear(pl_module.model.feat_dim, self.num_classes).to(pl_module.device)
            self.optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
        encoder.eval()
        train_acc = self._run_epoch(encoder, self.train_loader, pl_module.device, train=True)
        test_acc = self._run_epoch(encoder, self.test_loader, pl_module.device, train=False)
        encoder.train()
        pl_module.log_dict({"probe_train_acc": train_acc, "probe_test_acc": test_acc}, prog_bar=True)


class Pretraining(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.model = build_model(self.hparams)
        self.model.compile()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch["crops"], batch["coords"])["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        hp = self.hparams
        params = list(self.parameters())
        optimizer = torch.optim.AdamW([
            {"params": [p for p in params if p.ndim != 1], "weight_decay": hp.weight_decay},
            {"params": [p for p in params if p.ndim == 1], "weight_decay": 0.0},  # biases, norms
        ], lr=hp.lr)
        if hp.warmup_epochs == 0:
            return optimizer
        # linear warmup, then constant
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=hp.warmup_start_lr / hp.lr, total_iters=hp.warmup_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": warmup, "interval": "epoch"}}


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(output_dir, exist_ok=True)

    def loader(split, num_crops, shuffle=False, drop_last=False):
        dataset = TolkachESCA(args.data_path, split, num_crops, args.input_size, seed=args.seed)
        return make_loader(dataset, args.batch_size, args.num_workers, args.seed, shuffle, drop_last)

    train_loader = loader("train", args.num_crops, shuffle=True, drop_last=True)
    probe = OnlineProbe(loader("train", 1, shuffle=True), loader("test", 1),
                        num_classes=len(train_loader.dataset.classes))
    callbacks = [probe, ModelCheckpoint(dirpath=output_dir, every_n_epochs=args.save_every, save_last=True)]

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=[args.gpu_id],
                         precision=args.precision, logger=CSVLogger(output_dir, name="logs"),
                         callbacks=callbacks, default_root_dir=output_dir)
    trainer.fit(Pretraining(vars(args)), train_dataloaders=train_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
