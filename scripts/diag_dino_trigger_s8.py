"""Probe (read-only): does a ViT-S/8 @224 backbone restore trigger event
contrast on the native 28x28 patch grid (8px/cell, f_map=24.25)?

Context: ViT-S/14 @392 (28x28 grid) failed the adaptive trigger operating
point (0 clips everywhere; events 8-12% above calm-p99 vs MNV3 18-29%).

WEIGHT SUBSTITUTE (verified unobtainable otherwise): the official
DINOv2 ViT-S/8 checkpoint was REMOVED from dl.fbaipublicfiles.com (403
while dinov2_vits14 returns 200 — checked 2026-09-09), no HF/ModelScope
mirror exists, and the cached facebookresearch/dinov2 main-branch code
needs Python 3.10+. This probe therefore uses **DINOv1 ViT-S/8**
(facebook/dino-vits8, HF safetensors) — same native 28x28 grid and 8px
cells, testing the same granularity hypothesis, but DINOv1 features.
If this probe looks good, the DINOv2-S/8 full run still needs a weights
source (or the project accepts the v1 substitute).

The ViT is rebuilt standalone in plain torch; every shape is read from
the checkpoint itself, so there is no architecture guessing; a broken
forward cannot pass the sanity check (adjacent-frame pooled cosine on a
mostly-static rig) nor produce structured event contrast.

Usage: PYTHONPATH=. .venv/bin/python scripts/diag_dino_trigger_s8.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from cerebellum_cancel.core import SurpriseTrigger, TriggerConfig
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              maxcell_score)
from cerebellum_cancel.io import load_imu_csv

sys.path.insert(0, "scripts")
from train_downstream import seq_dirs  # noqa: E402

HF_REPO = "facebook/dino-vits8"
SIZE = 224                    # native 28x28 patch grid: no interpolation
BATCH = 64


# ------------------------------------------- standalone DINOv1 ViT-S/8
# HF transformers key layout -> fused standalone modules.


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_up = nn.Linear(dim, hidden)
        self.mlp_down = nn.Linear(hidden, dim)
        self.heads = heads

    def forward(self, x):
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.attn_qkv(h).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        o = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.attn_proj(o)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.norm2(x))))
        return x


class DinoV1_S8(nn.Module):
    def __init__(self, state: dict):
        super().__init__()
        pe = state["embeddings.position_embeddings"]   # (1, 1+gh*gw, C)
        self.gh = int(round((pe.shape[1] - 1) ** 0.5))
        self.dim = pe.shape[2]
        ks = state["embeddings.patch_embeddings.projection.weight"].shape[-1]
        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=ks, stride=ks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, pe.shape[1], self.dim))
        n_blocks = sum(1 for k in state if ".layernorm_before.weight" in k)
        hidden = state["encoder.layer.0.intermediate.dense.weight"].shape[0]
        self.blocks = nn.ModuleList(
            [Block(self.dim, 6, hidden) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(self.dim)
        fused = self._remap(state)
        self.load_state_dict(fused, strict=True)

    def _remap(self, state):
        """HF separate-qkv -> fused qkv; drop pooler/head keys."""
        out = {}
        for k, v in state.items():
            if k.startswith("pooler") or ".pooler" in k:
                continue
            out[k] = v
        for i in range(len(self.blocks)):
            p = f"encoder.layer.{i}.attention.attention."
            w = torch.cat([out.pop(p + n + ".weight") for n in ("query",
                                                                "key",
                                                                "value")])
            b = torch.cat([out.pop(p + n + ".bias") for n in ("query",
                                                              "key",
                                                              "value")])
            out[f"blocks.{i}.attn_qkv.weight"] = w
            out[f"blocks.{i}.attn_qkv.bias"] = b
            out[f"blocks.{i}.attn_proj.weight"] = out.pop(
                f"encoder.layer.{i}.attention.output.dense.weight")
            out[f"blocks.{i}.attn_proj.bias"] = out.pop(
                f"encoder.layer.{i}.attention.output.dense.bias")
            out[f"blocks.{i}.mlp_up.weight"] = out.pop(
                f"encoder.layer.{i}.intermediate.dense.weight")
            out[f"blocks.{i}.mlp_up.bias"] = out.pop(
                f"encoder.layer.{i}.intermediate.dense.bias")
            out[f"blocks.{i}.mlp_down.weight"] = out.pop(
                f"encoder.layer.{i}.output.dense.weight")
            out[f"blocks.{i}.mlp_down.bias"] = out.pop(
                f"encoder.layer.{i}.output.dense.bias")
            out[f"blocks.{i}.norm1.weight"] = out.pop(
                f"encoder.layer.{i}.layernorm_before.weight")
            out[f"blocks.{i}.norm1.bias"] = out.pop(
                f"encoder.layer.{i}.layernorm_before.bias")
            out[f"blocks.{i}.norm2.weight"] = out.pop(
                f"encoder.layer.{i}.layernorm_after.weight")
            out[f"blocks.{i}.norm2.bias"] = out.pop(
                f"encoder.layer.{i}.layernorm_after.bias")
        out["patch_embed.weight"] = out.pop(
            "embeddings.patch_embeddings.projection.weight")
        out["patch_embed.bias"] = out.pop(
            "embeddings.patch_embeddings.projection.bias")
        out["cls_token"] = out.pop("embeddings.cls_token")
        out["pos_embed"] = out.pop("embeddings.position_embeddings")
        out["norm.weight"] = out.pop("layernorm.weight")
        out["norm.bias"] = out.pop("layernorm.bias")
        return out

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        b = x.shape[0]
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"x_norm_patchtokens": x[:, 1:]}


def load_s8(device: str):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path = hf_hub_download(HF_REPO, "model.safetensors")
    state = load_file(path)
    model = DinoV1_S8(state).to(device).eval()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(),
                    T.Normalize(mean, std)])
    return model, tf


@torch.no_grad()
def extract_s8(model, tf, frames, device):
    outs = []
    for i in range(0, len(frames), BATCH):
        imgs = [tf(Image.open(p).convert("RGB")) for p in frames[i:i + BATCH]]
        x = torch.stack(imgs).to(device)
        toks = model.forward_features(x)["x_norm_patchtokens"]
        outs.append(toks.cpu().numpy().astype(np.float32))
    toks = np.concatenate(outs)                    # (n, 784, 384)
    return toks.reshape(len(toks), model.gh, model.gh, -1).transpose(0, 3, 1, 2)


# ------------------------------------------------------- trigger diagnostics


def rollout_score(fm, gyro):
    canc = FeatureMapCancel(CancelConfig(f_map=24.25 * fm.shape[-1] / 28.0))
    zs, sur, blocked = [], [], 0
    for t in range(len(fm)):
        out = canc.step(fm[t], gyro[t])
        zs.append(out["z"])
        sur.append(out["surprise"])
        if canc.n > canc.cfg.warmup and not out["learned"]:
            blocked += 1
    return maxcell_score(np.stack(zs)), np.array(sur), blocked


def trig_z(score, fps, n):
    cfg = TriggerConfig()
    inner = SurpriseTrigger(cfg, fps)
    mu, var, cnt = 0.0, 1.0, 0
    zs = np.zeros(n)
    for t in range(1, n):
        cnt += 1
        s = float(score[t])
        if cnt == 1:
            mu = s
        sd = float(np.sqrt(var)) + 1e-3 * (abs(mu) + 1.0)
        z = (s - mu) / sd
        zs[t] = z
        if cnt <= 30:
            d = s - mu
            mu += 0.02 * d
            var += 0.02 * (d * d - var)
            continue
        on = inner.update(t / fps, z, 1.0)
        if not on and z < cfg.threshold_k:
            d = s - mu
            mu += 0.02 * d
            var += 0.02 * (d * d - var)
    return zs, inner.finalize(n / fps)


def evaluate(fm, gyro, labels, fps):
    n = min(len(fm), len(gyro))
    fm, gyro = fm[:n], gyro[:n]
    score, sur, blocked = rollout_score(fm, gyro)
    ev = labels[:n] > 0
    calm_p99 = np.percentile(score[~ev], 99)
    z, clips = trig_z(score, fps, n)
    hi = np.where(z > 6)[0]
    runs = []
    for i in hi:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return {
        "C": fm.shape[1], "grid": fm.shape[-1],
        "score_mean": float(score.mean()), "score_sd": float(score.std()),
        "calm_p99": float(calm_p99),
        "ev_above_calm_p99": float((score[ev] > calm_p99).mean()),
        "surprise_mean": float(sur.mean()),
        "gate_blocked": blocked / n,
        "z_gt6": len(hi), "z_runs": [len(r) for r in runs],
        "z_ev_coinc": [bool(ev[i]) for i in hi],
        "clips": len(clips),
        # sanity: adjacent-frame pooled-token cosine on this static-ish rig
        "adj_cos": float(np.mean([
            float(np.dot(fm[i].mean(axis=(1, 2)), fm[i + 1].mean(axis=(1, 2)))
                  / (np.linalg.norm(fm[i].mean(axis=(1, 2)))
                     * np.linalg.norm(fm[i + 1].mean(axis=(1, 2))) + 1e-9))
            for i in range(0, min(n - 1, 200))])),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seqs = seq_dirs("data/dataset", "train")[:3]
    model, tf = load_s8(device)
    print(f"S/8 standalone @224 on {device}; native grid {model.gh}x{model.gh}, "
          f"f_map=24.25 (no interpolation)\n")
    rows = {}
    for seq in seqs:
        meta = json.load(open(f"{seq}/events.json"))
        fps = float(meta.get("fps", 30.0))
        labels = np.load(f"{seq}/labels.npy")
        gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                        dtype=np.float32)
        frames = sorted(glob.glob(f"{seq}/frames/*.png"))
        fm_s8 = extract_s8(model, tf, frames, device)
        fm_mnv3 = np.load(f"{seq}/featmaps.npy").astype(np.float32)
        fm_s14 = np.load(f"{seq}/featmaps_dino.npy").astype(np.float32)
        print(f"=== {seq} ===")
        for name, fm in (("MNV3", fm_mnv3), ("S14@392", fm_s14),
                         ("S8@224", fm_s8)):
            r = evaluate(fm, gyro, labels, fps)
            rows[(seq, name)] = r
            print(f"  {name:<8} C={r['C']} grid={r['grid']}x{r['grid']} "
                  f"adjCos={r['adj_cos']:.3f} "
                  f"score={r['score_mean']:.0f}±{r['score_sd']:.0f} "
                  f"calm_p99={r['calm_p99']:.0f} "
                  f"ev>calm_p99={r['ev_above_calm_p99']:.2f} "
                  f"surprise={r['surprise_mean']:.2f} "
                  f"blocked={r['gate_blocked']:.0%} "
                  f"z>6={r['z_gt6']} runs={r['z_runs']} "
                  f"ev={sum(r['z_ev_coinc'])}/{r['z_gt6']} "
                  f"clips={r['clips']}")
    print("\n=== 3-seq mean ===")
    print(f"{'backbone':<10}{'adjCos':>8}{'ev>calm_p99':>14}{'surprise':>10}"
          f"{'blocked':>9}{'z>6':>6}{'clips':>7}")
    for name in ("MNV3", "S14@392", "S8@224"):
        m = [rows[(s, name)] for s in seqs]
        print(f"{name:<10}{np.mean([r['adj_cos'] for r in m]):>8.3f}"
              f"{np.mean([r['ev_above_calm_p99'] for r in m]):>14.2f}"
              f"{np.mean([r['surprise_mean'] for r in m]):>10.2f}"
              f"{np.mean([r['gate_blocked'] for r in m]):>9.0%}"
              f"{np.mean([r['z_gt6'] for r in m]):>6.1f}"
              f"{np.mean([r['clips'] for r in m]):>7.2f}")


if __name__ == "__main__":
    main()
