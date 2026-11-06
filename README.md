# seq-JEPA-dense

Code for the paper

> **[Biomedical Self-Supervised Learning using Sequential and Action-Conditioned Prediction: A Proof of Concept in Histopathology](https://doi.org/10.5445/IR/1000198476)**<br>
> Linus Schilpp, Angelo Jovin Yamachui Sitcheu, Thomas Lautenschlager, Leonard Saur, Nils Friederich, Ralf Mikut, Stephan Allgeier<br>
> *Proc. 36. Workshop Computational Intelligence, Berlin, 2026*

Training and evaluation code for seq-JEPA-dense and the baselines
[SimCLR](https://proceedings.mlr.press/v119/chen20j.html) and
[seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html).

## Setup

Requires a CUDA GPU (we used a single NVIDIA RTX 3090) and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/lischilpp/seq-jepa-dense.git
cd seq-jepa-dense
uv sync
uv run python scripts/download_data.py      # saves to datasets/tolkach_esca
```

The script downloads the [PathoROB Tolkach ESCA](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tolkach_esca)
dataset from Hugging Face.

## Training

```bash
uv run python -m seq_jepa_dense.train --method simclr
uv run python -m seq_jepa_dense.train --method seqjepa
uv run python -m seq_jepa_dense.train --method seqjepa-dense
```

The defaults are the paper's settings. Each run saves its final checkpoint to `runs/<method>/last.ckpt`,
which the evaluation below uses. The training log also shows the accuracy of a linear probe trained
alongside, for monitoring only; the accuracies in the paper (Table 1) come from `eval_linear` below. Run `uv run python -m seq_jepa_dense.train --help`
for all options.

## Evaluation

**Linear-probe classification (Table 1).** Reports the test accuracy of the *res* and *agg* probes
(about a minute per model):

```bash
uv run python -m seq_jepa_dense.eval_linear --ckpt runs/seqjepa-dense/last.ckpt
```

**Held-out-crop retrieval (Table 2).** Reports within-image and cross-image Hit@1:

```bash
uv run python -m seq_jepa_dense.eval_retrieval \
    --ckpts runs/simclr/last.ckpt runs/seqjepa/last.ckpt runs/seqjepa-dense/last.ckpt
```

**UMAP projections (Figure 4):**

```bash
uv run python -m seq_jepa_dense.eval_umap --out umap \
    --ckpts runs/simclr/last.ckpt runs/seqjepa/last.ckpt runs/seqjepa-dense/last.ckpt
```

## Results

Single pretraining run per method, from the paper:

| Method         | *res* acc. (%) | *agg* acc. (%) | Within-image Hit@1 (%) | Cross-image Hit@1 (%) |
|----------------|---------------:|---------------:|-----------------------:|----------------------:|
| Chance         |                |                |                   20.0 |                 0.034 |
| SimCLR         |           91.7 |           93.3 |                      – |                  87.0 |
| seq-JEPA       |           86.6 |           88.3 |                   42.1 |                  80.6 |
| seq-JEPA-dense |           89.9 |           91.9 |                   41.8 |                  95.3 |

Results can differ slightly between runs and GPUs.

## Repository structure

```
src/seq_jepa_dense/
  data.py            dataset, train/test split, crop sampling
  models.py          SimCLR, seq-JEPA, seq-JEPA-dense
  common.py          checkpoint loading, representations used for evaluation
  train.py           pretraining
  eval_linear.py     Table 1
  eval_retrieval.py  Table 2
  eval_umap.py       Figure 4
  plotting.py        figure style for eval_umap
scripts/
  download_data.py
```

## Citation

```bibtex
@inproceedings{schilpp2026seqjepadense,
  author    = {Schilpp, Linus and Yamachui Sitcheu, Angelo Jovin and
               Lautenschlager, Thomas and Saur, Leonard and Friederich, Nils and
               Mikut, Ralf and Allgeier, Stephan},
  title     = {Biomedical Self-Supervised Learning using Sequential and Action-Conditioned Prediction: A Proof of Concept in Histopathology},
  booktitle = {Proceedings. 36. Workshop Computational Intelligence},
  editor    = {Schulte, Horst},
  address   = {Berlin},
  publisher = {{KIT Scientific Publishing}},
  pages     = {51--77},
  year      = {2026},
  doi       = {10.5445/IR/1000198476}
}
```

## License and acknowledgments

This code is released under the MIT license (see [LICENSE](LICENSE)). The models are adapted from the
official [seq-JEPA implementation](https://github.com/hafezgh/seq-jepa) by Ghaemi, Muller and Bakhtiari.

This work is supported by the Helmholtz Association under the program "Natural, Artificial and
Cognitive Information Processing (NACIP)" and the joint research school "Helmholtz Information & Data
Science School for Health (HIDSS4Health)".
