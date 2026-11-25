#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profillage d'inférence de la variante delta-rule « const » (sans clé ni requête).

État (par tête) : S_t = Σ_{s<t} β·(1−β)^(t−1−s)·v_s  (vecteur, pas matrice)
Lecture         : y_t = S_t / t   (t = position, division par le compte)
Mise à jour     : S ← (1−β)·S + β·v   — O(H·dh) par token, mémoire H·dh par couche.

Double garde-fou :
  (a) la forme parallèle locale (matrice A, même formule que c3b_controles.delta_fw
      branche « const ») est comparée à hn.forward avec la variante installée
      (c3b_controles.installer, delta="const") : écart attendu ~0 (mêmes ops float32) ;
  (b) la forme incrémentale (état) est comparée à la forme parallèle : écart max
      sur les logits < 2e-3 — même seuil que experiments/profiler_inf.py.

Mesures identiques à profiler_inf.py : V_gen à T0 ∈ {16, 48, 96} (150 tokens),
scaling mécanique T ∈ {256, 1024, 4096} (pos_emb neutralisée, 100 tokens),
mémoire (poids, état constant — ici H·dh par couche, sans Z ni matrice S).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_numpy as hn  # noqa: E402
from profiler_inf import charger_corpus  # noqa: E402


def beta_de(P, l):
    return (1.0 / (1.0 + np.exp(-P[f"b{l}.gamma"].astype(np.float32)))).astype(np.float32)


class GenerationConst:
    """Un token à la fois, type « dlindc » : état vectoriel à oubli exponentiel."""

    def __init__(self, P, types, n_head):
        assert all(t == "dlindc" for t in types), "ce profileur ne gère que dlindc"
        self.P, self.types, self.n_head = P, types, n_head
        self.D = P["tok_emb"].shape[1]
        self.dh = self.D // n_head
        self.reset()

    def reset(self):
        self.S = {l: np.zeros((self.n_head, self.dh), np.float32)
                  for l in range(len(self.types))}
        self.t = {l: 0 for l in range(len(self.types))}

    def pas(self, tok):
        P, D, H, dh = self.P, self.D, self.n_head, self.dh
        pos = min(self.t[0], P["pos_emb"].shape[0] - 1)
        x = P["tok_emb"][int(tok)][None, :] + P["pos_emb"][pos][None, :]
        for l in range(len(self.types)):
            h1, _, _ = hn.layer_norm_fw(x, P[f"b{l}.ln1.g"], P[f"b{l}.ln1.b"])
            qkv = h1 @ P[f"b{l}.qkv.W"] + P[f"b{l}.qkv.b"]
            v = qkv[:, 2 * D:].reshape(H, dh)
            b = beta_de(P, l)
            p = self.t[l]                                     # position 0-based de ce token
            y = (self.S[l] / max(p, 1)).reshape(1, D)         # y_t = S_t / t (y_0 = 0)
            self.S[l] = (1.0 - b)[:, None] * self.S[l] + b[:, None] * v
            self.t[l] = p + 1
            x = x + y @ P[f"b{l}.proj.W"] + P[f"b{l}.proj.b"]
            h2, _, _ = hn.layer_norm_fw(x, P[f"b{l}.ln2.g"], P[f"b{l}.ln2.b"])
            a = h2 @ P[f"b{l}.fc1.W"] + P[f"b{l}.fc1.b"]
            x = x + hn.gelu(a) @ P[f"b{l}.fc2.W"] + P[f"b{l}.fc2.b"]
        xf, _, _ = hn.layer_norm_fw(x, P["lnf.g"], P["lnf.b"])
        return (xf @ P["tok_emb"].T)[0]


def forward_const_parallele(P, cfg, idx):
    """Forme parallèle (matrice A) — même formule que c3b_controles.delta_fw « const »."""
    B, T = idx.shape
    D, H, V = cfg["d_model"], cfg["n_head"], cfg["vocab_size"]
    x = P["tok_emb"][idx] + P["pos_emb"][:T][None, :, :]
    t = np.arange(T)
    tt = np.maximum(t, 1)[None, :, None].astype(np.float32)          # (1,T,1)
    caches = []
    for l, _ in enumerate(cfg["types"]):
        h1, _, _ = hn.layer_norm_fw(x.reshape(B * T, D), P[f"b{l}.ln1.g"], P[f"b{l}.ln1.b"])
        h1 = h1.reshape(B, T, D)
        qkv = (h1.reshape(B * T, D) @ P[f"b{l}.qkv.W"] + P[f"b{l}.qkv.b"]).reshape(B, T, 3, D)
        v = qkv[:, :, 2].reshape(B, T, H, D // H).transpose(0, 2, 1, 3)   # (B,H,T,dh)
        b = beta_de(P, l)[:, None, None]                                  # (H,1,1)
        n = t[:, None] - 1 - t[None, :]
        pw = np.where(n >= 0, (1.0 - b) ** np.maximum(n, 0), 0.0).astype(np.float32)
        A = (b * pw / tt).astype(np.float32)                              # (H,T,T)
        y = A[None] @ v                                                   # (B,H,T,dh)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + y @ P[f"b{l}.proj.W"] + P[f"b{l}.proj.b"]
        h2, _, _ = hn.layer_norm_fw(x.reshape(B * T, D), P[f"b{l}.ln2.g"], P[f"b{l}.ln2.b"])
        a = (h2 @ P[f"b{l}.fc1.W"] + P[f"b{l}.fc1.b"]).reshape(B, T, 4 * D)
        x = x + hn.gelu(a) @ P[f"b{l}.fc2.W"] + P[f"b{l}.fc2.b"]
        caches.append(A)
    xf, _, _ = hn.layer_norm_fw(x.reshape(-1, D), P["lnf.g"], P["lnf.b"])
    return (xf @ P["tok_emb"].T).reshape(B, T, V)


def memoire_const(types, D, P, n_head=4):
    dh = D // n_head
    poids = sum(v.nbytes for v in P.values())
    etat = sum(n_head * dh * 4 for _ in types)      # S (H×dh) par couche, sans Z ni matrice
    return {"poids_octets": poids, "etat_lin_octets": etat, "kv_octets": 0}


def main():
    ap = argparse.ArgumentParser(description="Profilage dlind-const (état vectoriel)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--types", default="dlindc,dlindc")
    ap.add_argument("--data", default="data/fr1m/corpus.txt")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--out", default="results/profile_inf_dlindc-dlindc.json")
    a = ap.parse_args()
    P = {k: v for k, v in np.load(a.ckpt).items()}
    types = [t.strip() for t in a.types.split(",")]
    cfg = {"vocab_size": P["tok_emb"].shape[0], "d_model": a.d_model,
           "n_head": a.n_head, "types": ["dlind", "dlind"], "block_size": 96}
    ids, chars = charger_corpus(a.data)
    print(f"[PROFILEUR CONST] ckpt={a.ckpt}")

    # garde (a) : forme parallèle locale vs hn.forward installé « const »
    import c3b_controles as cc
    cc.installer("elu1", False, "const")
    try:
        ids_court = ids[300:396]
        ref = hn.forward(P, cfg, ids_court[None])[0]
        loc = forward_const_parallele(P, cfg, ids_court[None])
        ecart_a = float(np.abs(ref - loc).max())
    finally:
        cc.desinstaller()
    print(f"garde (a) parallèle locale vs hn.forward « const » : {ecart_a:.2e}")

    # garde (b) : incrémental vs forme parallèle locale
    g = GenerationConst(P, types, a.n_head)
    diffs = []
    for t in range(96):
        diffs.append(g.pas(int(ids_court[t])))
    ecart_b = float(np.abs(np.array(diffs) - loc[0]).max())
    print(f"garde (b) incrémental vs parallèle : {ecart_b:.2e} (seuil 2e-3)")
    ok = ecart_b < 2e-3
    if ecart_a > 1e-4:
        print("⚠ garde (a) hors tolerance — les mesures sont invalidées")
        ok = False

    res = {"types": types, "ckpt": a.ckpt,
           "garde_a_parallele_vs_hn": ecart_a, "garde_b_incremental_vs_parallele": ecart_b,
           "garde_ok": ok}

    if not ok:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        sys.exit(1)

    # V_gen dans la distribution : T0 ∈ {16, 48, 96}
    res["v_gen_dans_distribution"] = {}
    for T0 in (16, 48, 96):
        g.reset()
        t0 = time.time()
        for i in range(T0):
            g.pas(int(ids[i]))
        tok = int(ids[T0 - 1])
        for _ in range(a.n_gen):
            tok = int(np.argmax(g.pas(tok)))
        dt = time.time() - t0
        v = a.n_gen / dt
        res["v_gen_dans_distribution"][str(T0)] = {"tok_s": round(v, 1),
                                                   "prefill_tok_s": round(T0 / dt, 1)}
        print(f"V_gen @T0={T0:2d} : {v:.1f} tok/s (prefill {T0 / dt:.0f} tok/s)")

    # scaling mécanique : absorption de T tokens puis 100 générés (pos_emb neutralisée)
    res["scaling_mecanisme"] = {}
    for T in (256, 1024, 4096):
        g.reset()
        t0 = time.time()
        for i in range(T):
            g.pas(int(ids[i % len(ids)]))
        prefill = T / (time.time() - t0)
        tok = int(ids[T - 1])
        t0 = time.time()
        for _ in range(100):
            tok = int(np.argmax(g.pas(tok)))
        v = 100 / (time.time() - t0)
        res["scaling_mecanisme"][str(T)] = {"prefill_tok_s": round(prefill, 1),
                                            "v_gen_tok_s": round(v, 1),
                                            "kv_octets": 0}
        print(f"scaling T={T:5d} : prefill {prefill:.0f} tok/s | V_gen {v:.1f} tok/s | KV 0 Ko")

    res["memoire"] = memoire_const(types, a.d_model, P)
    m = res["memoire"]
    print(f"mémoire : poids {m['poids_octets'] / 1024:.0f} Ko | état {m['etat_lin_octets'] / 1024:.1f} Ko | KV 0 Ko")
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Écrit : {a.out}")


if __name__ == "__main__":
    main()
