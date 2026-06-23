"""
Training entry point.

Two modes selected by the `mode` field in the config:

  mode: baseline   -> objective 1.
                      One shared encoder + ONE classifier, trained on the union
                      of all source data (no domain alignment). Evaluated
                      zero-shot on the target to expose the domain shift.

  mode: msda       -> objectives 2-3.
                      Shared encoder + per-source classifiers + adversarial
                      domain discriminator (GRL). Weighted ensemble produces the
                      target prediction and the per-source influence ratio.

Run:
    python -m src.training.train --config experiments/configs/baseline.yaml
    python -m src.training.train --config experiments/configs/model_v1.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.common import load_config, set_seed, accuracy
from src.data.datasets import build_loaders, InfiniteSampler, DOMAIN_IDS
from src.data.class_mapping import NUM_CLASSES
from src.models.multisource_da import MultiSourceDA, SharedEncoder, ClassifierHead
from src.models.grl import grl_lambda


# --------------------------------------------------------------------------- #
# Baseline: union of sources, single head, no alignment (objective 1)
# --------------------------------------------------------------------------- #
class BaselineModel(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=256):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, embed_dim)
        self.classifier = ClassifierHead(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def train_baseline(cfg, device):
    source_loaders, _, target_eval, feat_dim = build_loaders(cfg)
    model = BaselineModel(feat_dim, NUM_CLASSES, cfg["embed_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Merge all source batches each step.
    samplers = {d: InfiniteSampler(l) for d, l in source_loaders.items()}
    steps = cfg["steps"]
    model.train()
    for step in range(steps):
        feats, labels = [], []
        for d in samplers:
            x, y, _ = samplers[d].next()
            feats.append(x)
            labels.append(y)
        x = torch.cat(feats).to(device)
        y = torch.cat(labels).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % cfg["log_every"] == 0:
            print(f"[baseline] step {step+1}/{steps} loss={loss.item():.4f} "
                  f"train_acc={accuracy(logits, y):.3f}")

    save_checkpoint(model, cfg, kind="baseline")
    evaluate_baseline(model, target_eval, device)


@torch.no_grad()
def evaluate_baseline(model, target_eval, device):
    model.eval()
    correct = total = 0
    for x, y, _ in target_eval:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
    print(f"[baseline] TARGET zero-shot accuracy = {correct/total:.4f}")


# --------------------------------------------------------------------------- #
# Multi-source DA (objectives 2-3)
# --------------------------------------------------------------------------- #
def train_msda(cfg, device):
    source_loaders, target_loader, target_eval, feat_dim = build_loaders(cfg)
    sources = cfg["sources"]
    model = MultiSourceDA(
        in_dim=feat_dim,
        num_classes=NUM_CLASSES,
        source_domains=sources,
        embed_dim=cfg["embed_dim"],
        num_domains=3,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    src_samplers = {d: InfiniteSampler(l) for d, l in source_loaders.items()}
    tgt_sampler = InfiniteSampler(target_loader)
    steps = cfg["steps"]
    adv_w = cfg["adversarial_weight"]
    temperature = cfg.get("ensemble_temperature", 0.1)
    drop_prob = cfg.get("drop_prob", 0.0)   # extra: simulate incomplete batches
    import random as _random

    model.train()
    for step in range(steps):
        lambd = grl_lambda(step, steps) * adv_w
        cls_loss = 0.0
        dom_loss = 0.0

        # --- source-drop simulation (extra objective) ---
        # With prob drop_prob, drop one source this step, as if its data never
        # arrived ("DataLost"). We never drop all sources at once.
        active_sources = list(sources)
        if drop_prob > 0 and len(sources) > 1 and _random.random() < drop_prob:
            dropped = _random.choice(sources)
            active_sources = [s for s in sources if s != dropped]

        # --- supervised + adversarial for each active source ---
        for dom in active_sources:
            x, y, dom_id = src_samplers[dom].next()
            x, y, dom_id = x.to(device), y.to(device), dom_id.to(device)
            z = model.encode(x)
            logits = model.classify_source(z, dom)
            cls_loss = cls_loss + F.cross_entropy(logits, y)
            d_logits = model.discriminate(z, lambd)
            dom_loss = dom_loss + F.cross_entropy(d_logits, dom_id)
            model.update_centroid(dom, z.detach())

        # --- adversarial for target (no class labels) ---
        xt, _, dom_id_t = tgt_sampler.next()
        xt, dom_id_t = xt.to(device), dom_id_t.to(device)
        zt = model.encode(xt)
        dt_logits = model.discriminate(zt, lambd)
        dom_loss = dom_loss + F.cross_entropy(dt_logits, dom_id_t)

        loss = cls_loss + dom_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % cfg["log_every"] == 0:
            print(f"[msda] step {step+1}/{steps} lambda={lambd:.3f} "
                  f"cls={cls_loss.item():.3f} dom={dom_loss.item():.3f}")

        if (step + 1) % cfg["eval_every"] == 0:
            evaluate_msda(model, target_eval, sources, device, temperature)
            model.train()

    save_checkpoint(model, cfg, kind="msda")
    return evaluate_msda(model, target_eval, sources, device, temperature)


@torch.no_grad()
def evaluate_msda(model, target_eval, sources, device, temperature=0.1):
    model.eval()
    correct = total = 0
    weight_accum = torch.zeros(len(sources), device=device)
    nbatches = 0
    # per-class counts for macro-accuracy
    from src.data.class_mapping import NUM_CLASSES
    per_class_correct = torch.zeros(NUM_CLASSES)
    per_class_total = torch.zeros(NUM_CLASSES)
    for x, y, _ in target_eval:
        x, y = x.to(device), y.to(device)
        logits, weights = model.ensemble_predict(x, temperature=temperature, return_weights=True)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
        weight_accum += weights
        nbatches += 1
        for t, p in zip(y.cpu(), preds.cpu()):
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1
    avg_w = (weight_accum / nbatches).tolist()
    ratio = {s: round(w, 4) for s, w in zip(sources, avg_w)}
    acc = correct / total
    # macro-accuracy: mean of per-class accuracies (classes weighted equally)
    valid = per_class_total > 0
    macro = (per_class_correct[valid] / per_class_total[valid]).mean().item()
    print(f"[msda] TARGET accuracy = {acc:.4f} | macro-acc = {macro:.4f} | "
          f"source influence = {ratio}")
    return acc, ratio


# --------------------------------------------------------------------------- #
def save_checkpoint(model, cfg, kind):
    out = Path(cfg.get("ckpt_dir", "experiments/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}_{cfg.get('run_name', 'run')}.pt"
    torch.save(model.state_dict(), path)
    print(f"[ckpt] saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] mode={cfg['mode']} device={device}")

    if cfg["mode"] == "baseline":
        train_baseline(cfg, device)
    elif cfg["mode"] == "msda":
        train_msda(cfg, device)
    else:
        raise ValueError(f"Unknown mode: {cfg['mode']}")


if __name__ == "__main__":
    main()
