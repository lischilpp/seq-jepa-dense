"""Linear-probe tissue classification (paper, Table 1).

    python -m seq_jepa_dense.eval_linear --ckpt runs/seqjepa-dense/last.ckpt

Trains two linear probes on the frozen model and reports their test accuracy
after the final epoch:
  res: encoder representation of a single crop
  agg: representation aggregated over N crops (mean encoder representation for
       SimCLR, z_AGG for seq-JEPA, h_N for seq-JEPA-dense)

Crops are sampled once per image, augmented for the training split and
unaugmented for the test split; the probes are trained on the cached
representations.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from seq_jepa_dense.common import METHODS, embed, get_device, load_model
from seq_jepa_dense.data import TolkachESCA, make_loader

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint written by train.py")
    p.add_argument("--data-path", default="datasets/tolkach_esca")
    p.add_argument("--num-context", type=int, default=4, help="crops N for the agg probe")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def extract(model, loader, device):
    """res and agg representations and labels of every image in `loader`."""
    feats, labels = {"res": [], "agg": []}, []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = embed(model, batch["crops"].to(device), batch["coords"].to(device))
        for name in feats:
            feats[name].append(out[name].float())
        labels.append(batch["label"].to(device))
    return {name: torch.cat(f) for name, f in feats.items()}, torch.cat(labels)


def accuracy(probe, feats, labels):
    """Top-1 accuracy in percent."""
    metric = MulticlassAccuracy(probe.out_features, average="micro").to(labels.device)
    with torch.no_grad(), torch.autocast(labels.device.type, dtype=torch.bfloat16):
        metric.update(probe(feats), labels)
    return 100.0 * metric.compute().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.gpu_id)
    model, hp = load_model(args.ckpt, device)

    def features(split, shuffle):
        dataset = TolkachESCA(args.data_path, split, args.num_context + 1, hp["input_size"], seed=args.seed)
        loader = make_loader(dataset, args.batch_size, args.num_workers, args.seed, shuffle=shuffle)
        return extract(model, loader, device)

    train_feats, train_labels = features("train", shuffle=True)
    test_feats, test_labels = features("test", shuffle=False)
    num_classes = int(train_labels.max()) + 1

    probes = nn.ModuleDict({name: nn.Linear(f.shape[1], num_classes)
                            for name, f in train_feats.items()}).to(device)
    optimizer = torch.optim.Adam(probes.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_labels), device=device, generator=generator)
        for idx in perm.split(args.batch_size):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = sum(F.cross_entropy(probe(train_feats[name][idx]).float(), train_labels[idx])
                           for name, probe in probes.items())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = {name: accuracy(probe, train_feats[name], train_labels) for name, probe in probes.items()}
            print(f"epoch {epoch + 1}/{args.epochs}  train acc: res {acc['res']:.2f}  agg {acc['agg']:.2f}")

    acc = {name: accuracy(probe, test_feats[name], test_labels) for name, probe in probes.items()}
    print(f"\n{METHODS[hp['method']]} test top-1 accuracy after {args.epochs} epochs "
          f"(N={args.num_context}): res {acc['res']:.1f}%  agg {acc['agg']:.1f}%")


if __name__ == "__main__":
    main()
