"""Model construction, checkpoint loading and the representations used for evaluation."""
import torch

from seq_jepa_dense.models import SeqJEPA, SimCLR

METHODS = {"simclr": "SimCLR", "seqjepa": "seq-JEPA", "seqjepa-dense": "seq-JEPA-dense"}


def build_model(hp):
    """Model for the training hyperparameters `hp` (see train.py)."""
    if hp["method"] == "simclr":
        return SimCLR(hp["temperature"])
    return SeqJEPA(dense=hp["method"] == "seqjepa-dense", action_emb_dim=hp["action_emb_dim"],
                   num_heads=hp["num_heads"], num_layers=hp["num_layers"],
                   pred_hidden=hp["pred_hidden"])


def load_model(ckpt_path, device):
    """Frozen model in eval mode from a train.py checkpoint, and its hyperparameters.
    The architecture is rebuilt from the hyperparameters stored in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    model = build_model(hp)
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()})
    model.requires_grad_(False)
    print(f"Loaded {METHODS[hp['method']]} from {ckpt_path}")
    return model.to(device).eval(), hp


def get_device(gpu_id):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(model, crops, coords):
    """Representations of a batch of crop sequences, crops: (B, N + 1, C, H, W).
    The first N crops are the context, crop N+1 is the held-out target.

      res:     encoder representation of the first crop
      agg:     representation aggregated over the N context crops: their mean
               encoder representation for SimCLR, z_AGG for seq-JEPA, h_N for
               seq-JEPA-dense
      query:   retrieval query for the target: for SimCLR its agg, for the
               seq-JEPA variants the (final-step) prediction for the target
      target:  encoder representation of the target
      context: encoder representations of the context crops, (B, N, D)
    """
    B, S = crops.shape[:2]
    z = model.encoder(crops.flatten(0, 1)).view(B, S, -1)
    out = {"res": z[:, 0], "target": z[:, -1], "context": z[:, :-1]}
    if isinstance(model, SimCLR):
        out["agg"] = out["query"] = z[:, :-1].mean(dim=1)
    else:
        pred = model(crops, coords)
        out["agg"], out["query"] = pred["agg"], pred["pred"]
    return out
