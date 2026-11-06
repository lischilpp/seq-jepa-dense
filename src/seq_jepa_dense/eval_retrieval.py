"""Held-out-crop retrieval, Hit@1 (paper, Table 2).

    python -m seq_jepa_dense.eval_retrieval --ckpts runs/simclr/last.ckpt runs/seqjepa/last.ckpt runs/seqjepa-dense/last.ckpt

For every test image, N context crops and one held-out target crop are
sampled. The query is the predictor output for the action leading to the
target (seq-JEPA variants; final-step prediction for seq-JEPA-dense) or the
mean encoder representation of the context crops (SimCLR). Candidates are
ranked by cosine similarity to the query.

  within-image: target vs. the image's own N context crops (chance 1/(N+1));
                not reported for SimCLR, whose query averages these candidates
  cross-image:  target vs. the held-out targets of all test images (chance 1/M)
"""
import argparse

import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalHitRate
from tqdm import tqdm

from seq_jepa_dense.common import METHODS, embed, get_device, load_model
from seq_jepa_dense.data import TolkachESCA, make_loader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoints written by train.py")
    p.add_argument("--data-path", default="datasets/tolkach_esca")
    p.add_argument("--num-context", type=int, default=4, help="context crops N")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def hit_rate(scores, labels, k=1):
    """Hit@k over queries (rows) of a (queries, candidates) score matrix, with
    boolean labels marking the correct candidate."""
    M, N = scores.shape
    indexes = torch.arange(M).unsqueeze(1).expand(M, N).flatten()
    return RetrievalHitRate(top_k=k)(scores.flatten(), labels.flatten(), indexes=indexes).item()


def hit_at_1(query, target, context):
    """Within-image and cross-image Hit@1 in percent; query, target: (M, D), context: (M, N, D)."""
    sim = pairwise_cosine_similarity(query, target)                           # (M, M)
    cross = hit_rate(sim, torch.eye(len(sim), dtype=torch.bool))

    # candidates: the image's N context crops and its target (last column)
    ctx_sim = F.cosine_similarity(query.unsqueeze(1), context, dim=-1)        # (M, N)
    scores = torch.cat([ctx_sim, sim.diagonal().unsqueeze(1)], dim=1)
    labels = torch.zeros_like(scores, dtype=torch.bool)
    labels[:, -1] = True
    within = hit_rate(scores, labels)
    return 100 * within, 100 * cross


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    models = [load_model(path, device) for path in args.ckpts]
    input_sizes = {hp["input_size"] for _, hp in models}
    assert len(input_sizes) == 1, "all models must use the same input size"

    dataset = TolkachESCA(args.data_path, "test", args.num_context + 1, input_sizes.pop(), seed=args.seed)
    loader = make_loader(dataset, args.batch_size, args.num_workers, args.seed)

    feats = [{"query": [], "target": [], "context": []} for _ in models]
    for batch in tqdm(loader, desc="extracting"):
        crops, coords = batch["crops"].to(device), batch["coords"].to(device)
        for (model, _), buf in zip(models, feats):
            out = embed(model, crops, coords)
            for k in buf:
                buf[k].append(out[k].float().cpu())

    M, N = len(dataset), args.num_context
    print(f"\n{'Method':<16}{'Within-image (%)':>18}{'Cross-image (%)':>18}")
    print(f"{'Chance':<16}{100 / (N + 1):>18.1f}{100 / M:>18.3f}")
    for (_, hp), buf in zip(models, feats):
        within, cross = hit_at_1(*(torch.cat(buf[k]) for k in ("query", "target", "context")))
        within = "-" if hp["method"] == "simclr" else f"{within:.1f}"
        print(f"{METHODS[hp['method']]:<16}{within:>18}{cross:>18.1f}")


if __name__ == "__main__":
    main()
