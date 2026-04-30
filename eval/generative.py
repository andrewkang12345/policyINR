"""Generative / predictive metrics, numerically robust.

Two code paths:
  * continuous actions -> MSE / NMSE / median squared error
  * discrete actions   -> top-1 accuracy / NLL

The selection is based on `next_action`'s dtype (long -> discrete).
Both paths write into a common summary schema so the aggregator can
tell the two apart by which fields are present.
"""

from __future__ import annotations

from typing import Dict, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def generative_metrics(model, loader, device: str,
                        max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    # peek the first batch to decide continuous vs discrete
    it = iter(loader)
    try:
        first = next(it)
    except StopIteration:
        return {"gen_mse": float("nan"), "gen_nmse": float("nan"),
                "gen_median_se": float("nan"), "n_gen_samples": 0,
                "finite_fraction": 0.0}
    is_discrete = first["next_action"].dtype in (torch.int64, torch.long, torch.int32)

    def batches():
        yield first
        for b in it:
            yield b

    if is_discrete:
        correct = 0
        total = 0
        nll_sum = 0.0
        n_classes = None
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}
        for bi, batch in enumerate(batches()):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(batch)
            target = batch["next_action"].long()
            pred = out.get("pred", None)
            if pred is None:
                # models that don't return logits from forward — fall back
                # to predict_action; we only get argmax, no NLL.
                a_hat = model.predict_action(batch).long()
                correct += (a_hat == target).sum().item()
                total += target.numel()
                continue
            if pred.dim() == target.dim():
                # models returning argmax ids (e.g. diffusion at discrete)
                a_hat = pred.long()
                correct += (a_hat == target).sum().item()
                total += target.numel()
                continue
            # logits: (B, K)
            log_probs = F.log_softmax(pred.float(), dim=-1)
            nll_sum += float(F.nll_loss(log_probs, target, reduction="sum").item())
            a_hat = log_probs.argmax(dim=-1)
            correct += int((a_hat == target).sum().item())
            total += int(target.numel())
            n_classes = pred.shape[-1]
            # per-class breakdown
            for c in torch.unique(target).tolist():
                m = target == c
                per_class_correct[c] = per_class_correct.get(c, 0) + int((a_hat[m] == c).sum().item())
                per_class_total[c] = per_class_total.get(c, 0) + int(m.sum().item())
        acc = correct / max(1, total)
        nll = nll_sum / max(1, total) if total else float("nan")
        return {
            "gen_acc": float(acc),
            "gen_nll": float(nll),
            "gen_n_classes": int(n_classes) if n_classes else None,
            "gen_per_class_acc": {int(c): per_class_correct[c] / per_class_total[c]
                                  for c in per_class_total if per_class_total[c]},
            "n_gen_samples": int(total),
            "finite_fraction": 1.0,
            "gen_mse": float("nan"),
            "gen_nmse": float("nan"),
            "gen_median_se": float("nan"),
        }

    # continuous
    all_err: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []
    count = 0
    finite_samples = 0
    total_samples = 0
    per_dim_se = None
    action_dim = None
    for bi, batch in enumerate(batches()):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model.predict_action(batch)
        target = batch["next_action"]
        finite_mask = torch.isfinite(pred).all(dim=-1)
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
        err = (pred - target).pow(2)
        all_err.append(err.cpu())
        all_tgt.append(target.cpu())
        count += err.shape[0]
        if per_dim_se is None:
            action_dim = err.shape[-1]
            per_dim_se = torch.zeros(action_dim)
        per_dim_se += err.sum(dim=0).cpu()
        finite_samples += int(finite_mask.sum().item())
        total_samples += int(finite_mask.numel())
    if count == 0:
        return {"gen_mse": float("nan"), "gen_rmse": float("nan"),
                "gen_nmse": float("nan"), "gen_median_se": float("nan"),
                "n_gen_samples": 0, "finite_fraction": 0.0}

    err = torch.cat(all_err, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    per_sample_se = err.mean(dim=-1)
    mse = float(err.mean().item())
    median_se = float(per_sample_se.median().item())
    target_var = float(tgt.var(dim=0, unbiased=False).mean().item()) + 1e-6
    nmse = mse / target_var
    nmse_per_dim = (err.mean(dim=0) / (tgt.var(dim=0, unbiased=False) + 1e-6)).tolist()
    return {
        "gen_mse": mse,
        "gen_rmse": math.sqrt(max(mse, 0.0)),
        "gen_nmse": float(nmse),
        "gen_median_se": float(median_se),
        "gen_nmse_per_dim": [float(x) for x in nmse_per_dim],
        "target_var": float(target_var),
        "n_gen_samples": int(count),
        "finite_fraction": float(finite_samples / max(1, total_samples)),
    }
