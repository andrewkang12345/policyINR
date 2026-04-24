"""Generic Trainer used by all models.

Every model returns `dict(loss=..., ...)` from `forward(batch)`, and
every dataset batch has the same key set (see data/base.py). That's
the whole contract the Trainer relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import time

import torch
from torch.amp import autocast, GradScaler

from utils.logging import get_logger, JSONLLogger


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 50
    val_every_epoch: int = 1
    device: str = "cuda"
    amp: bool = True
    early_stop_patience: int = 0


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: TrainConfig,
        output_dir: Path,
        run_name: str = "run",
    ):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.logger = get_logger(f"inr.train.{run_name}")
        self.metric_log = JSONLLogger(self.output_dir / "metrics.jsonl")
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scaler = GradScaler("cuda", enabled=(cfg.amp and cfg.device.startswith("cuda")))
        self._amp_device = "cuda" if cfg.device.startswith("cuda") else "cpu"

    def _move(self, batch: Dict[str, torch.Tensor]):
        return {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items()}

    def _step(self, batch):
        batch = self._move(batch)
        with autocast(self._amp_device, enabled=self.scaler.is_enabled()):
            out = self.model(batch)
            loss = out["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: (v.item() if torch.is_tensor(v) and v.ndim == 0 else v)
                for k, v in out.items() if k != "pred" and k != "pred_eps"}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total = {"loss": 0.0, "count": 0}
        for batch in self.val_loader:
            batch = self._move(batch)
            with autocast(self._amp_device, enabled=self.scaler.is_enabled()):
                out = self.model(batch)
            bs = batch["next_action"].shape[0]
            total["loss"] += out["loss"].item() * bs
            total["count"] += bs
        self.model.train()
        if total["count"] == 0:
            return {}
        return {"val_loss": total["loss"] / total["count"]}

    def fit(self) -> Dict[str, Any]:
        self.model.train()
        best_val = float("inf")
        best_epoch = 0
        patience = 0
        global_step = 0
        t0 = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            running = 0.0
            n = 0
            for bi, batch in enumerate(self.train_loader):
                step_out = self._step(batch)
                running += step_out.get("loss", 0.0)
                n += 1
                global_step += 1
                if global_step % self.cfg.log_every == 0:
                    self.logger.info(f"ep {epoch} step {global_step} loss {running / max(1, n):.4f}")

            train_loss = running / max(1, n)
            val = self.validate() if (epoch % self.cfg.val_every_epoch == 0) else {}
            record = {"epoch": epoch, "train_loss": train_loss, **val,
                      "elapsed_s": time.time() - t0}
            self.metric_log.log(record)
            self.logger.info(f"epoch {epoch}: {record}")

            vloss = val.get("val_loss", float("inf"))
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_epoch = epoch
                patience = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch},
                           self.output_dir / "best.pt")
            else:
                patience += 1
                if self.cfg.early_stop_patience and patience >= self.cfg.early_stop_patience:
                    self.logger.info(f"early stop at epoch {epoch}")
                    break

        torch.save({"model": self.model.state_dict(), "epoch": self.cfg.epochs},
                   self.output_dir / "last.pt")
        summary = {"best_val_loss": best_val, "best_epoch": best_epoch,
                   "final_train_loss": train_loss, "total_epochs": epoch,
                   "wallclock_s": time.time() - t0}
        return summary
