"""CA-EviNet (JOINT): Conflict-aware Evidential Network for two mitral-valve tasks.

Pipeline (matches the paper's Method section and Algorithm 1):
  per view: T frames -> shared ViT-S/16 -> temporal Transformer -> view feature f_j
  for the etiology task:
     task-specific evidential head -> e_j^tau -> Dirichlet(alpha_j) -> p_j, u_j
     conflict-aware RAL: inter-view conflict delta_j -> reliability weight w_j
                         -> aggregated evidence -> prob^tau, uncertainty U^tau
  auxiliary: masked-attention pooling of {f_j} -> f_agg -> MLP -> M measurements

Ablation switches (C.ABLATE): notemporal | noview | noconflict | noevidential | noquant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


class FrameViT(nn.Module):
    """2D ViT that maps one grayscale frame to a feature vector.
    - "vit_s_16": ImageNet ViT-S/16 (timm, D=384). Setting used for the paper.
    - "vit_b_16": ImageNet torchvision ViT-B/16 (D=768).
    - "small":    tiny MONAI ViT from scratch (fast crash-testing only).
    `self.last_tokens` holds the patch tokens (N,P,hidden) for Grad-CAM.
    """
    def __init__(self):
        super().__init__()
        self.last_tokens = None
        if C.BACKBONE == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.kind = "tv"
            self.tv = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.tv.heads = nn.Identity()
            self.hidden = 768
        elif C.BACKBONE == "vit_s_16":
            import timm                                    # ImageNet ViT-S/16 (~22M)
            self.kind = "timm"
            self.timm = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
            self.hidden = 384
        else:
            from monai.networks.nets import ViT
            self.kind = "monai"
            cfg = C.vit_cfg()
            self.hidden = cfg["hidden_size"]
            self.vit = ViT(
                in_channels=C.IN_CH, img_size=(C.IMG_SIZE, C.IMG_SIZE),
                patch_size=(C.PATCH_SIZE, C.PATCH_SIZE), spatial_dims=2,
                hidden_size=cfg["hidden_size"], mlp_dim=cfg["mlp_dim"],
                num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
                classification=False, dropout_rate=0.0,
            )

    def forward(self, x):                       # x: (N,1,H,W)
        if self.kind == "tv":
            x = x.repeat(1, 3, 1, 1)
            t = self.tv._process_input(x)
            cls = self.tv.class_token.expand(x.shape[0], -1, -1)
            t = self.tv.encoder(torch.cat([cls, t], dim=1))
            self.last_tokens = t[:, 1:]
            return t[:, 0]
        if self.kind == "timm":
            x = x.repeat(1, 3, 1, 1)
            t = self.timm.forward_features(x)   # (N, 1+P, 384) tokens incl. CLS
            self.last_tokens = t[:, 1:]
            return t[:, 0]
        out = self.vit(x)
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        self.last_tokens = tokens
        return tokens.mean(dim=1)


class TemporalEncoder(nn.Module):
    """Transformer over the T frame features of a view -> single view feature."""
    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, C.T, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=C.TEMPORAL_HEADS, dim_feedforward=dim * 2,
            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=C.TEMPORAL_LAYERS)

    def forward(self, x):                       # x: (B, T, dim)
        x = x + self.pos
        x = self.enc(x)
        return x.mean(dim=1)                     # (B, dim)


class CAEviNet(nn.Module):
    def __init__(self, num_classes=C.NUM_CLASSES, m=C.M, gamma=C.GAMMA):
        super().__init__()
        self.K = num_classes
        self.V = len(C.VIEWS)
        self.gamma = gamma
        self.ablate = C.ABLATE
        self.frame_vit = FrameViT()
        dim = self.frame_vit.hidden
        self.temporal = TemporalEncoder(dim)
        # one evidential (or, for noevidential, logit) head per task
        self.heads = nn.ModuleDict({t["name"]: nn.Linear(dim, num_classes) for t in C.TASKS})
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(getattr(C, "DROPOUT", 0.0))
        # masked-attention pooling for the quantification branch
        self.q_query = nn.Parameter(torch.zeros(dim)); nn.init.trunc_normal_(self.q_query, std=0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.quant_head = nn.Sequential(
            nn.Linear(dim, C.QUANT_HIDDEN), nn.GELU(),
            nn.Dropout(getattr(C, "DROPOUT", 0.0)),
            nn.Linear(C.QUANT_HIDDEN, m))
        if getattr(C, "FREEZE_BACKBONE", False):
            for p in self.frame_vit.parameters():
                p.requires_grad = False

    def encode(self, views):
        """views: (B,V,T,1,H,W) -> view features f: (B,V,dim)."""
        B, V, T = views.shape[:3]
        x = views.reshape(B * V * T, C.IN_CH, C.IMG_SIZE, C.IMG_SIZE)
        g = self.frame_vit(x)                                # (B*V*T, dim)
        g = g.reshape(B * V, T, -1)                          # (B*V, T, dim)
        if self.ablate == "notemporal":
            f = g.mean(dim=1)                                # ablation: mean over frames
        else:
            f = self.temporal(g)                             # (B*V, dim)
        return f.reshape(B, V, -1)                            # (B,V,dim)

    def _ral(self, f, head, avail):
        """Conflict-aware Reliable Aggregation for one task.
        f: (B,V,dim); head: nn.Linear; avail a_j: (B,V) in {0,1}. Returns dict."""
        B, V, _ = f.shape
        e = self.softplus(head(self.drop(f)))                # (B,V,K) evidence
        alpha = e + 1.0
        S = alpha.sum(-1)                                    # (B,V)
        p = alpha / S.unsqueeze(-1)                          # (B,V,K)
        u = self.K / S                                      # (B,V)
        # inter-view conflict: availability-masked mean TV distance to other views
        dmat = 0.5 * (p.unsqueeze(2) - p.unsqueeze(1)).abs().sum(-1)   # (B,V,V)
        offdiag = (1.0 - torch.eye(V, device=f.device)).unsqueeze(0)   # (1,V,V)
        wpair = avail.unsqueeze(1) * offdiag                 # (B,V,V) other-view availability
        delta = (wpair * dmat).sum(-1) / wpair.sum(-1).clamp(min=1.0)  # (B,V)
        if self.ablate == "noconflict":                      # uniform average of available views
            r = avail.clone()
        else:
            r = avail * (1.0 - u) * torch.exp(-self.gamma * delta)     # (B,V)
        w = r / (r.sum(dim=1, keepdim=True) + C.RAL_EPS)     # (B,V)
        e_agg = (w.unsqueeze(-1) * e).sum(dim=1)             # (B,K)
        alpha_agg = e_agg + 1.0
        S_agg = alpha_agg.sum(-1, keepdim=True)
        prob = alpha_agg / S_agg                            # (B,K)
        U = self.K / S_agg.squeeze(-1)                      # (B,)
        return {"alpha_views": alpha, "u_views": u, "conflict": delta, "weights": w,
                "alpha": alpha_agg, "prob": prob, "uncertainty": U}

    def _ce_fuse(self, f, head, avail):
        """Ablation 'noevidential': treat head output as logits, mean-fuse views,
        softmax. No Dirichlet / no RAL. Uncertainty = 1 - max prob (for metrics)."""
        B, V, _ = f.shape
        logits_v = head(self.drop(f))                        # (B,V,K)
        denom = avail.sum(dim=1, keepdim=True).clamp(min=1.0)
        logits = (avail.unsqueeze(-1) * logits_v).sum(dim=1) / denom  # (B,K)
        prob = F.softmax(logits, dim=-1)
        U = 1.0 - prob.max(dim=-1).values
        w = avail / avail.sum(dim=1, keepdim=True).clamp(min=1.0)
        return {"logits": logits, "logits_views": logits_v, "prob": prob,
                "uncertainty": U, "weights": w,
                "conflict": torch.zeros(B, V, device=f.device),
                "u_views": torch.zeros(B, V, device=f.device)}

    def _quant_pool(self, f, avail):
        """Masked-attention pooling over available views -> (B,dim)."""
        s = (torch.tanh(self.q_proj(f)) * self.q_query).sum(-1)       # (B,V)
        s = s.masked_fill(avail < 0.5, float("-inf"))
        beta = F.softmax(s, dim=1)                                    # (B,V)
        return (beta.unsqueeze(-1) * f).sum(dim=1)                    # (B,dim)

    def forward(self, views, avail=None):
        f = self.encode(views)                               # (B,V,dim)
        B, V, dim = f.shape
        if avail is None:
            avail = torch.ones(B, V, device=f.device)        # all views present
        out = {"feat": f, "avail": avail, "tasks": {}}
        for t in C.TASKS:
            head = self.heads[t["name"]]
            if self.ablate == "noevidential":
                out["tasks"][t["name"]] = self._ce_fuse(f, head, avail)
            else:
                out["tasks"][t["name"]] = self._ral(f, head, avail)
        f_agg = self._quant_pool(f, avail)                   # (B,dim)
        out["quant"] = self.quant_head(f_agg)                # (B,M)
        return out
