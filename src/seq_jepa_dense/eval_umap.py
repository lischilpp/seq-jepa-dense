"""UMAP projections of the res and agg representations (paper, Figure 4).

    python -m seq_jepa_dense.eval_umap --ckpts runs/simclr/last.ckpt runs/seqjepa/last.ckpt runs/seqjepa-dense/last.ckpt

One column per checkpoint, in the given order; rows are res (N=1) and agg
(N context crops). Uses all 16,300 images with unaugmented, seeded crops.
"""
import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from umap import UMAP

import seq_jepa_dense.plotting as pf
from seq_jepa_dense.common import METHODS, embed, get_device, load_model
from seq_jepa_dense.data import TolkachESCA, make_loader
from seq_jepa_dense.plotting import FONTSIZE, TEXTWIDTH_IN

PANEL_GAP_IN, ROW_GAP_IN = 0.05, 0.05
LEFT_IN, RIGHT_IN, BOTTOM_IN = 0.20, 0.02, 0.02
TITLE_IN = 0.20       # column titles above the top panels
LEGEND_PAD_IN = 0.10  # between the legend and the column titles


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoints written by train.py")
    p.add_argument("--data-path", default="datasets/tolkach_esca")
    p.add_argument("--num-context", type=int, default=4, help="context crops N for agg")
    p.add_argument("--max-samples", type=int, default=None, help="random subset, for quick runs")
    p.add_argument("--out", default="umap", help="output path without extension")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300, help="resolution of the rasterized scatter")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def extract(models, loader, device):
    """res and agg representations of every model on identical crops."""
    feats = [{"res": [], "agg": []} for _ in models]
    labels = []
    for batch in tqdm(loader, desc="extracting"):
        crops, coords = batch["crops"].to(device), batch["coords"].to(device)
        for (model, _), buf in zip(models, feats):
            out = embed(model, crops, coords)
            for k in buf:
                buf[k].append(out[k].float().cpu().numpy())
        labels.extend(batch["label"].tolist())
    return [{k: np.concatenate(v) for k, v in buf.items()} for buf in feats], np.array(labels)


def legend_layout(classes, colors, handlelength=1.2, handletextpad=0.4, columnspacing=1.0):
    """Number of legend columns that fits the figure width (rows balanced, e.g.
    3+3 rather than 4+2), and the legend's measured height in inches."""
    probe = pf.probe_figure(TEXTWIDTH_IN, 2)
    text_w = max(pf.measure(probe, c, FONTSIZE) for c in classes)
    pt = FONTSIZE / 72.0
    entry_w = (handlelength + handletextpad + columnspacing) * pt + text_w
    fits = max(1, min(len(classes), int((TEXTWIDTH_IN - 0.04) // entry_w)))
    nrow = -(-len(classes) // fits)
    ncol = -(-len(classes) // nrow)
    leg = probe.legend(handles=[mpatches.Patch(color=colors[k], label=k) for k in classes],
                       loc="upper center", ncol=ncol, fontsize=FONTSIZE, frameon=False,
                       handlelength=handlelength, handleheight=1.2,
                       columnspacing=columnspacing, handletextpad=handletextpad)
    height = leg.get_window_extent(pf.renderer(probe)).height / probe.dpi
    plt.close(probe)
    return ncol, height + LEGEND_PAD_IN


def scatter(ax, emb, labels, colors, title):
    # Rasterized points keep the PDF small; text and axes stay vector.
    ax.scatter(emb[:, 0], emb[:, 1], c=[colors[l] for l in labels], s=1.5,
               alpha=0.5, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")  # UMAP axes must share a scale
    if title:
        ax.set_title(title, fontsize=FONTSIZE, pad=4)
    pf.style(ax)
    for spine in ax.spines.values():  # thin frame to separate the tightly spaced panels
        spine.set_visible(True)
        spine.set_edgecolor("0.8")
        spine.set_linewidth(0.6)


def plot_grid(panels, classes, num_context, out, formats, dpi):
    """panels: (title, res embedding, agg embedding, class names) per column."""
    colors = pf.palette(classes)
    ncol, legend_h = legend_layout(classes, colors)
    top_in = TITLE_IN + legend_h
    n_cols = len(panels)
    panel = (TEXTWIDTH_IN - LEFT_IN - RIGHT_IN - (n_cols - 1) * PANEL_GAP_IN) / n_cols
    fig_h = 2 * panel + ROW_GAP_IN + top_in + BOTTOM_IN

    fig, axes = plt.subplots(
        2, n_cols, figsize=(TEXTWIDTH_IN, fig_h), squeeze=False,
        gridspec_kw=dict(wspace=PANEL_GAP_IN / panel, hspace=ROW_GAP_IN / panel,
                         **pf.margins(TEXTWIDTH_IN, fig_h, LEFT_IN, RIGHT_IN, top_in, BOTTOM_IN)))
    row_labels = [r"$\mathit{res}$ ($N{=}1$)", rf"$\mathit{{agg}}$ ($N{{=}}{num_context}$)"]
    for col, (title, emb_res, emb_agg, labels) in enumerate(panels):
        for row, emb in enumerate([emb_res, emb_agg]):
            ax = axes[row][col]
            scatter(ax, emb, labels, colors, title if row == 0 else "")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=FONTSIZE, labelpad=3)

    fig.legend(handles=[mpatches.Patch(color=colors[k], label=k) for k in classes],
               loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.998), fontsize=FONTSIZE,
               frameon=False, handlelength=1.2, handleheight=1.2, columnspacing=1.0,
               handletextpad=0.4)
    pf.save(fig, out, formats, dpi)


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    models = [load_model(path, device) for path in args.ckpts]
    input_sizes = {hp["input_size"] for _, hp in models}
    assert len(input_sizes) == 1, "all models must use the same input size"

    dataset = TolkachESCA(args.data_path, "all", args.num_context + 1, input_sizes.pop(), seed=args.seed)
    classes = dataset.classes
    if args.max_samples is not None and args.max_samples < len(dataset):
        keep = np.random.RandomState(args.seed).choice(len(dataset), args.max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, sorted(keep.tolist()))
    loader = make_loader(dataset, args.batch_size, args.num_workers, args.seed)

    feats, labels = extract(models, loader, device)
    names = [classes[l] for l in labels]
    panels = []
    for (_, hp), f in zip(models, feats):
        title = METHODS[hp["method"]]
        print(f"Fitting UMAP for {title} ...")
        res, agg = (UMAP(n_neighbors=30, min_dist=0.1, random_state=args.seed).fit_transform(f[k])
                    for k in ("res", "agg"))
        panels.append((title, res, agg, names))
    plot_grid(panels, classes, args.num_context, args.out, args.formats, args.dpi)


if __name__ == "__main__":
    main()
