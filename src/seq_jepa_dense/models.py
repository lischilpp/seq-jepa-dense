"""SimCLR, seq-JEPA and seq-JEPA-dense (paper, Section 3.3).

Adapted from the official seq-JEPA implementation (https://github.com/hafezgh/seq-jepa).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ACTION_DIM = 3  # (du, dv, dzeta)


def resnet18_encoder():
    """Randomly initialized ResNet-18 without its classification head (512-d output)."""
    encoder = torchvision.models.resnet18(weights=None, zero_init_residual=True)
    encoder.fc = nn.Identity()
    return encoder


class SeqJEPA(nn.Module):
    """seq-JEPA (dense=False) and seq-JEPA-dense (dense=True), with online targets.

    seq-JEPA aggregates the N observed crops, each carrying its outgoing action,
    into the [AGG] token and predicts the representation of the held-out crop
    N+1 from it and the final action.
    seq-JEPA-dense uses a causal mask, each token carries its incoming action,
    and the predictor forecasts the next crop's representation at every step.
    """

    def __init__(self, dense, action_emb_dim=128, num_heads=4, num_layers=3, pred_hidden=1024):
        super().__init__()
        self.dense = dense
        self.encoder = resnet18_encoder()
        self.feat_dim = 512
        self.action_emb_dim = action_emb_dim
        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_DIM, action_emb_dim, bias=False),
            nn.BatchNorm1d(action_emb_dim, affine=False))
        self.emb_dim = self.feat_dim + action_emb_dim
        self.predictor = nn.Sequential(
            nn.Linear(self.emb_dim + action_emb_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, self.feat_dim))
        # [AGG] token; seq-JEPA-dense uses it as the [BOS] token
        self.agg_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        layer = nn.TransformerEncoderLayer(self.emb_dim, num_heads, dim_feedforward=4 * self.emb_dim,
                                           batch_first=True)
        self.aggregator = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(self.emb_dim),
                                                enable_nested_tensor=False)

    def forward(self, crops, coords):
        """crops: (B, S, C, H, W), coords: (B, S, 3) with S = N + 1.
        The first N crops are observed; crop N+1 is held out as the final target.

        Returns a dict with
          loss: mean negative cosine similarity between predictions and targets,
          agg:  aggregate representation (z_AGG, or h_N for seq-JEPA-dense),
          pred: predicted representation of crop N+1.
        """
        B, S = crops.shape[:2]
        N = S - 1
        z = self.encoder(crops.flatten(0, 1)).view(B, S, -1)
        actions = coords[:, 1:] - coords[:, :-1]  # (B, N, 3), actions[:, i]: crop i -> i+1

        if self.dense:
            token_actions = F.pad(actions[:, :-1], (0, 0, 1, 0))  # incoming, zero for crop 1
            pred_actions = actions                                # outgoing, at every step
        else:
            token_actions = F.pad(actions[:, :-1], (0, 0, 0, 1))  # outgoing, zero for crop N
            pred_actions = actions[:, -1:]                        # final action only
        
        # One BatchNorm call for token and predictor actions, so they share batch statistics.
        emb = self.action_proj(torch.cat([token_actions, pred_actions], 1).flatten(0, 1))
        emb = emb.view(B, -1, self.action_emb_dim)
        token_emb, pred_emb = emb[:, :N].clone(), emb[:, N:]
        token_emb[:, 0 if self.dense else -1] = 0.0  # the missing action is a zero vector

        x = torch.cat([z[:, :N], token_emb], dim=-1)
        x = torch.cat([self.agg_token.expand(B, -1, -1), x], dim=1)  # (B, N + 1, emb_dim)

        if self.dense:
            mask = nn.Transformer.generate_square_subsequent_mask(N + 1, device=x.device, dtype=x.dtype)
            h = self.aggregator(x, mask=mask, is_causal=True)[:, 1:]  # h_1, ..., h_N
            agg = h[:, -1]
            pred = self.predictor(torch.cat([h, pred_emb], dim=-1).flatten(0, 1)).view(B, N, -1)
            target = z[:, 1:]
        else:
            agg = self.aggregator(x)[:, 0]
            pred = self.predictor(torch.cat([agg, pred_emb[:, 0]], dim=-1)).unsqueeze(1)
            target = z[:, -1:]

        loss = 1 - F.cosine_similarity(pred, target.detach(), dim=-1).mean()
        return {"loss": loss, "agg": agg, "pred": pred[:, -1]}


def ntxent_logits(features, batch_size, temperature):
    """NT-Xent logits and labels for features (2B, D) of B positive pairs, as in
    the seq-JEPA SimCLR baseline: for every view, the logit of its positive
    comes first, followed by those of its 2B - 2 negatives."""
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return logits, labels


class SimCLR(nn.Module):
    """SimCLR with the projection head of the seq-JEPA SimCLR baseline."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.encoder = resnet18_encoder()
        self.feat_dim = 512
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, affine=False))

    def forward(self, crops, coords=None):
        """crops: (B, 2, C, H, W), a positive pair per image. coords are unused."""
        # The two views go through the network separately, so BatchNorm
        # statistics are computed per view.
        y1 = self.projector(self.encoder(crops[:, 0]))
        y2 = self.projector(self.encoder(crops[:, 1]))
        logits, labels = ntxent_logits(torch.cat([y1, y2], dim=0), y1.shape[0], self.temperature)
        return {"loss": F.cross_entropy(logits, labels)}
