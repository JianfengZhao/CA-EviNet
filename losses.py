"""Losses for the JOINT CA-EviNet.
- Evidential loss: Bayes-risk MSE under the Dirichlet + annealed (capped) KL to
  Dir(1) (Sensoy et al., NeurIPS 2018). Applied to each view (light, weight 0.3)
  and to the aggregated Dirichlet, summed over both tasks.
- Quantification loss: masked Huber regression over the M measurements.
Ablations: 'noview' sets the per-view weight to 0; 'noevidential' replaces the
evidential terms with softmax cross-entropy on the fused logits.
"""
import torch
import torch.nn.functional as F
import config as C


def kl_to_uniform_dirichlet(alpha):
    """KL( Dir(alpha) || Dir(1) ).  alpha: (B,K) -> (B,1)."""
    K = alpha.shape[-1]
    beta = torch.ones((1, K), device=alpha.device, dtype=alpha.dtype)  # Dir(1)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg_S = torch.digamma(S_alpha)
    dg_a = torch.digamma(alpha)
    kl = ((alpha - beta) * (dg_a - dg_S)).sum(dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def edl_bayes_risk_mse(alpha, y_onehot):
    """Expected squared error under Dir(alpha).  (B,1)."""
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = ((y_onehot - p) ** 2).sum(dim=1, keepdim=True)
    var = (p * (1.0 - p) / (S + 1.0)).sum(dim=1, keepdim=True)
    return err + var


def evidential_loss(alpha, y, lam_t):
    """Full evidential loss for one Dirichlet head.  y: (B,) long."""
    K = alpha.shape[-1]
    y1 = F.one_hot(y, num_classes=K).float()
    mse = edl_bayes_risk_mse(alpha, y1)
    alpha_tilde = y1 + (1.0 - y1) * alpha          # keep only misleading evidence
    kl = kl_to_uniform_dirichlet(alpha_tilde)
    return (mse + lam_t * kl).mean()


def quant_loss(pred, target, mask):
    """Masked Huber loss over the M measurements.  all (B,M)."""
    h = F.huber_loss(pred, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return h.sum() / denom


def task_loss(task_out, y, avail, lam_t):
    """Evidential (or, for noevidential, CE) loss for a single task."""
    if C.ABLATE == "noevidential":
        l_agg = F.cross_entropy(task_out["logits"], y)
        l_view = torch.zeros((), device=y.device)
        if C.PER_VIEW_W > 0 and C.ABLATE != "noview":
            lv = task_out["logits_views"]                       # (B,V,K)
            B, V, K = lv.shape
            per = F.cross_entropy(lv.reshape(B * V, K),
                                  y.unsqueeze(1).expand(B, V).reshape(B * V),
                                  reduction="none").reshape(B, V)
            l_view = (avail * per).sum() / avail.sum().clamp(min=1.0)
        return l_agg + (0.0 if C.ABLATE == "noview" else C.PER_VIEW_W) * l_view, \
            {"agg": float(l_agg.item()), "view": float(l_view.item() if torch.is_tensor(l_view) else 0.0)}
    # evidential path
    l_agg = evidential_loss(task_out["alpha"], y, lam_t)
    l_view = torch.zeros((), device=y.device)
    if C.ABLATE != "noview" and C.PER_VIEW_W > 0:
        alpha_v = task_out["alpha_views"]                       # (B,V,K)
        V = alpha_v.shape[1]
        acc = 0.0
        for v in range(V):
            # availability-masked per-view evidential loss
            av = avail[:, v]
            if av.sum() < 1:
                continue
            lv = evidential_loss(alpha_v[:, v, :], y, lam_t)
            acc = acc + lv
        l_view = acc / max(1, V)
    w = 0.0 if C.ABLATE == "noview" else C.PER_VIEW_W
    return l_agg + w * l_view, {"agg": float(l_agg.item()),
                                "view": float(l_view.item() if torch.is_tensor(l_view) else 0.0)}


def total_loss(out, batch, lam_t, beta):
    """Sum the per-task classification losses and the quantification loss."""
    avail = out["avail"]
    total = torch.zeros((), device=avail.device)
    parts = {}
    for t in C.TASKS:
        y = batch[f"label_{t['name']}"]
        lt, pt = task_loss(out["tasks"][t["name"]], y, avail, lam_t)
        total = total + lt
        parts[f"l_{t['name']}_agg"] = pt["agg"]
        parts[f"l_{t['name']}_view"] = pt["view"]
    l_q = quant_loss(out["quant"], batch["quant"], batch["quant_mask"])
    beta_eff = 0.0 if C.ABLATE == "noquant" else beta
    total = total + beta_eff * l_q
    parts["l_quant"] = float(l_q.item())
    return total, parts
