#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 IMANE_DEEP — Expérience C1 : ratio linéaire/softmax par couche
=============================================================================
Question C1 : combien de couches softmax suffisent-elles pour récupérer la
qualité d'un transformer, tout en gardant le profil mémoire du linéaire ?

Chaque couche est au choix :
  - "lin"  : attention linéaire à état (O(T), zéro KV-cache, cf. RWKV/GLA)
  - "smax" : attention softmax causale (O(T²), KV-cache)

--types "lin,lin"   = 0 % softmax  (baseline : results/baseline_lin_lin.csv)
--types "lin,smax"  = hybride 50 % (LE cycle C1)
--types "smax,smax" = 100 % softmax (baseline : results/baseline_smax_smax.csv)

Discipline : --check-grads OBLIGATOIRE avant tout entraînement (protocole §4).
=============================================================================
"""
import argparse
import json
import math
import os
import time

import numpy as np

EPS_LN, EPS_DEN = 1e-5, 1e-6


# ---------- briques partagées (identiques aux baselines) ----------

def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

def d_gelu(x):
    c = math.sqrt(2.0 / math.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (c * (1.0 + 3.0 * 0.044715 * x ** 2))

def layer_norm_fw(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS_LN)
    xhat = (x - mu) * rstd
    return g * xhat + b, xhat, rstd

def layer_norm_bw(dy, xhat, rstd, g):
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True)) * rstd
    return dx, dg, db

def phi(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 0.0))) + 1.0

def d_phi(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))

def revcumsum(x, axis=1):
    return np.cumsum(np.flip(x, axis=axis), axis=axis).take(
        range(x.shape[axis] - 1, -1, -1), axis=axis)

def softmax_dernier(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


# ---------- attention softmax (par tête) ----------

def att_softmax_fw(q, k, v, scale, mask):
    att = (q @ k.transpose(0, 1, 3, 2)) * scale
    att = np.where(mask[None, None], att, att.dtype.type(-1e10))
    att = softmax_dernier(att)
    return att @ v, {"q": q, "k": k, "v": v, "att": att}

def att_softmax_bw(dy, c, scale, mask):
    q, k, v, att = c["q"], c["k"], c["v"], c["att"]
    datt = dy @ v.transpose(0, 1, 3, 2)
    dv = att.transpose(0, 1, 3, 2) @ dy
    ds = att * (datt - (datt * att).sum(-1, keepdims=True))
    ds = np.where(mask[None, None], ds, ds.dtype.type(0))
    dq = (ds @ k) * scale
    dk = (ds.transpose(0, 1, 3, 2) @ q) * scale
    return dq, dk, dv


# ---------- attention linéaire à état (par tête) ----------

def att_lin_fw(q, k, v):
    Q, K = phi(q), phi(k)
    KV = np.einsum("bhti,bhtj->bhtij", K, v)
    S = np.cumsum(KV, axis=2)
    Z = np.cumsum(K, axis=2)
    Sp = np.zeros_like(S); Sp[:, :, 1:] = S[:, :, :-1]
    Zp = np.zeros_like(Z); Zp[:, :, 1:] = Z[:, :, :-1]
    num = np.einsum("bhti,bhtij->bhtj", Q, Sp)
    den = np.einsum("bhti,bhti->bht", Q, Zp) + EPS_DEN
    y = num / den[..., None]
    return y, {"q": q, "k": k, "v": v, "Q": Q, "K": K, "Sp": Sp, "Zp": Zp, "num": num, "den": den}

def att_lin_bw(dy, c):
    Q, K, v = c["Q"], c["K"], c["v"]
    Sp, Zp, num, den = c["Sp"], c["Zp"], c["num"], c["den"]
    dnum = dy / den[..., None]
    dden = -(dy * num).sum(-1) / (den ** 2)
    dQ = np.einsum("bhtj,bhtij->bhti", dnum, Sp) + dden[..., None] * Zp
    dSp = np.einsum("bhti,bhtj->bhtij", Q, dnum)
    dZp = Q * dden[..., None]
    dS = np.zeros_like(dSp); dS[:, :, :-1] = dSp[:, :, 1:]
    dZ = np.zeros_like(dZp); dZ[:, :, :-1] = dZp[:, :, 1:]
    dKV = revcumsum(dS, axis=2)
    dZsum = revcumsum(dZ, axis=2)
    dK = np.einsum("bhtij,bhtj->bhti", dKV, v) + dZsum
    dV = np.einsum("bhtij,bhti->bhtj", dKV, K)
    return dQ, dK, dV


# ---------- bloc (type par couche) ----------

def bloc_fw(P, l, typ, x, D, H, scale, mask):
    dh = D // H
    B, T, _ = x.shape
    N = B * T
    c = {"B": B, "T": T, "typ": typ}
    h1, xhat1, rstd1 = layer_norm_fw(x.reshape(N, D), P[f"b{l}.ln1.g"], P[f"b{l}.ln1.b"])
    h1 = h1.reshape(B, T, D)
    qkv = (h1.reshape(N, D) @ P[f"b{l}.qkv.W"] + P[f"b{l}.qkv.b"]).reshape(B, T, 3, D)
    q = qkv[:, :, 0].reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    y, att_c = att_lin_fw(q, k, v) if typ == "lin" else att_softmax_fw(q, k, v, scale, mask)
    c["att"], c["q"], c["k"] = att_c, q, k
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    c["y"] = y
    proj = (y.reshape(N, D) @ P[f"b{l}.proj.W"] + P[f"b{l}.proj.b"]).reshape(B, T, D)
    x_mid = x + proj
    h2, xhat2, rstd2 = layer_norm_fw(x_mid.reshape(N, D), P[f"b{l}.ln2.g"], P[f"b{l}.ln2.b"])
    h2 = h2.reshape(B, T, D)
    a = (h2.reshape(N, D) @ P[f"b{l}.fc1.W"] + P[f"b{l}.fc1.b"]).reshape(B, T, 4 * D)
    m = gelu(a)
    out = (m.reshape(N, 4 * D) @ P[f"b{l}.fc2.W"] + P[f"b{l}.fc2.b"]).reshape(B, T, D)
    c["h1"], c["h2"], c["a"], c["m"] = h1, h2, a, m
    c["xhat1"], c["rstd1"], c["xhat2"], c["rstd2"] = xhat1, rstd1, xhat2, rstd2
    return x_mid + out, c

def bloc_bw(P, l, typ, dx, c, grads, D, H, scale, mask):
    dx = dx.reshape(c["B"], c["T"], D)
    B, T, N = c["B"], c["T"], c["B"] * c["T"]
    dx_mid = dx.copy(); dout = dx
    dm = dout @ P[f"b{l}.fc2.W"].T
    grads[f"b{l}.fc2.W"] += c["m"].reshape(N, 4 * D).T @ dout.reshape(N, D)
    grads[f"b{l}.fc2.b"] += dout.reshape(N, D).sum(0)
    da = dm * d_gelu(c["a"])
    dh2 = da @ P[f"b{l}.fc1.W"].T
    grads[f"b{l}.fc1.W"] += c["h2"].reshape(N, D).T @ da.reshape(N, 4 * D)
    grads[f"b{l}.fc1.b"] += da.reshape(N, 4 * D).sum(0)
    dln2, dg2, db2 = layer_norm_bw(dh2.reshape(N, D), c["xhat2"], c["rstd2"], P[f"b{l}.ln2.g"])
    grads[f"b{l}.ln2.g"] += dg2; grads[f"b{l}.ln2.b"] += db2
    dx_mid += dln2.reshape(B, T, D)
    dproj = dx_mid
    dy = (dproj.reshape(N, D) @ P[f"b{l}.proj.W"].T).reshape(B, T, D)
    grads[f"b{l}.proj.W"] += c["y"].reshape(N, D).T @ dproj.reshape(N, D)
    grads[f"b{l}.proj.b"] += dproj.reshape(N, D).sum(0)
    dyh = dy.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
    if typ == "lin":
        dq, dk, dv = att_lin_bw(dyh, c["att"])
        dq = dq * d_phi(c["q"]); dk = dk * d_phi(c["k"])
    else:
        dq, dk, dv = att_softmax_bw(dyh, c["att"], scale, mask)
    dq2 = dq.transpose(0, 2, 1, 3).reshape(B, T, D)
    dk2 = dk.transpose(0, 2, 1, 3).reshape(B, T, D)
    dv2 = dv.transpose(0, 2, 1, 3).reshape(B, T, D)
    dqkv = np.concatenate([dq2[:, :, None, :], dk2[:, :, None, :], dv2[:, :, None, :]], axis=2).reshape(B, T, 3 * D)
    dh1 = dqkv.reshape(N, 3 * D) @ P[f"b{l}.qkv.W"].T
    grads[f"b{l}.qkv.W"] += c["h1"].reshape(N, D).T @ dqkv.reshape(N, 3 * D)
    grads[f"b{l}.qkv.b"] += dqkv.reshape(N, 3 * D).sum(0)
    dln1, dg1, db1 = layer_norm_bw(dh1.reshape(N, D), c["xhat1"], c["rstd1"], P[f"b{l}.ln1.g"])
    grads[f"b{l}.ln1.g"] += dg1; grads[f"b{l}.ln1.b"] += db1
    return dx_mid + dln1.reshape(B, T, D)


# ---------- modèle ----------

def init_params(V, cfg, rng, dtype=np.float32):
    P, L, D = {}, len(cfg["types"]), cfg["d_model"]
    std, rstd = 0.02, 1.0 / math.sqrt(2.0 * max(1, L))
    P["tok_emb"] = (rng.normal(0, std, (V, D))).astype(dtype)
    P["pos_emb"] = (rng.normal(0, std, (cfg["block_size"], D))).astype(dtype)
    for l in range(L):
        P[f"b{l}.qkv.W"] = (rng.normal(0, std, (D, 3 * D))).astype(dtype)
        P[f"b{l}.qkv.b"] = np.zeros(3 * D, dtype)
        P[f"b{l}.proj.W"] = (rng.normal(0, std * rstd, (D, D))).astype(dtype)
        P[f"b{l}.proj.b"] = np.zeros(D, dtype)
        P[f"b{l}.fc1.W"] = (rng.normal(0, std, (D, 4 * D))).astype(dtype)
        P[f"b{l}.fc1.b"] = np.zeros(4 * D, dtype)
        P[f"b{l}.fc2.W"] = (rng.normal(0, std * rstd, (4 * D, D))).astype(dtype)
        P[f"b{l}.fc2.b"] = np.zeros(D, dtype)
        P[f"b{l}.ln1.g"] = np.ones(D, dtype); P[f"b{l}.ln1.b"] = np.zeros(D, dtype)
        P[f"b{l}.ln2.g"] = np.ones(D, dtype); P[f"b{l}.ln2.b"] = np.zeros(D, dtype)
    P["lnf.g"] = np.ones(D, dtype); P["lnf.b"] = np.zeros(D, dtype)
    return P

def forward(P, cfg, idx, targets=None):
    B, T = idx.shape
    D, H, V = cfg["d_model"], cfg["n_head"], cfg["vocab_size"]
    scale = 1.0 / math.sqrt(D // H)
    mask = np.tril(np.ones((T, T), dtype=bool))
    x = P["tok_emb"][idx] + P["pos_emb"][:T][None, :, :]
    caches = []
    for l, typ in enumerate(cfg["types"]):
        x, c = bloc_fw(P, l, typ, x, D, H, scale, mask)
        caches.append(c)
    xf, xhatf, rstdf = layer_norm_fw(x.reshape(-1, D), P["lnf.g"], P["lnf.b"])
    logits = (xf @ P["tok_emb"].T).reshape(B, T, V)
    cache = {"xf": xf, "xhatf": xhatf, "rstdf": rstdf, "caches": caches, "idx": idx}
    perte = None
    if targets is not None:
        N = B * T
        p = softmax_dernier(logits.reshape(N, V))
        t = targets.reshape(N)
        perte = -np.log(p[np.arange(N), t] + 1e-12).mean()
        cache["p"], cache["t"], cache["N"] = p, t, N
    return logits, perte, cache

def backward(P, cfg, cache, targets):
    B, T = cache["idx"].shape
    D, V = cfg["d_model"], cfg["vocab_size"]
    N = cache["N"]
    scale = 1.0 / math.sqrt(D // cfg["n_head"])
    mask = np.tril(np.ones((T, T), dtype=bool))
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    dlogits = cache["p"].copy()
    dlogits[np.arange(N), cache["t"]] -= 1.0
    dlogits /= N
    xf = cache["xf"]
    grads["tok_emb"] += dlogits.T @ xf
    dxf = (dlogits @ P["tok_emb"]).reshape(B, T, D)
    dx, dg, db = layer_norm_bw(dxf.reshape(N, D), cache["xhatf"], cache["rstdf"], P["lnf.g"])
    grads["lnf.g"] += dg; grads["lnf.b"] += db
    for l in reversed(range(len(cfg["types"]))):
        dx = bloc_bw(P, l, cfg["types"][l], dx, cache["caches"][l], grads, D, cfg["n_head"], scale, mask)
    np.add.at(grads["tok_emb"], cache["idx"].reshape(-1), dx.reshape(-1, D))
    grads["pos_emb"][:T] += dx.sum(0)
    return grads


# ---------- optimiseur / données / vérification ----------

class AdamW:
    def __init__(self, P, wd=0.05, b1=0.9, b2=0.95, eps=1e-8):
        self.P, self.wd, self.b1, self.b2, self.eps = P, wd, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0
    def step(self, grads, lr):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in self.P.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            decay = self.wd if (p.ndim >= 2 and not k.startswith("ln")) else 0.0
            p -= lr * ((self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps) + decay * p)

def clip_gradient(grads, max_norm=1.0):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        f = max_norm / (norm + 1e-6)
        for k in grads: grads[k] *= f
    return norm

def taux_apprentissage(step, peak, warmup, total, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

def charger_donnees(chemin):
    with open(chemin, encoding="utf-8") as f:
        texte = f.read()
    chars = sorted(set(texte))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in texte], dtype=np.int32)
    return texte, ids, stoi, chars

def prendre_batch(data, B, T, rng):
    ix = rng.integers(0, len(data) - T - 1, size=B)
    return np.stack([data[i:i + T] for i in ix]), np.stack([data[i + 1:i + T + 1] for i in ix])

def verifier_gradients(args):
    print("[GARDE-FOU] vérification des gradients (float64, types mixtes lin+smax)...")
    texte, ids, _, itos = charger_donnees(args.data)
    ids = ids[:300]
    cfg = {"vocab_size": len(itos), "d_model": 16, "n_head": 2,
           "types": ["lin", "smax"], "block_size": 12}
    rng = np.random.default_rng(0)
    P = init_params(len(itos), cfg, rng, dtype=np.float64)
    x, y = prendre_batch(ids, B=2, T=cfg["block_size"], rng=rng)

    def perte_seule(P_):
        return forward(P_, cfg, x, y)[1]

    _, perte, cache = forward(P, cfg, x, y)
    grads = backward(P, cfg, cache, y)
    rng2 = np.random.default_rng(1)
    eps = 1e-5
    pire = 0.0
    for k in P:
        flat, gflat = P[k].reshape(-1), grads[k].reshape(-1)
        for i in rng2.integers(0, flat.size, size=min(4, flat.size)):
            sauve = flat[i]
            flat[i] = sauve + eps; lp = perte_seule(P)
            flat[i] = sauve - eps; lm = perte_seule(P)
            flat[i] = sauve
            num = (lp - lm) / (2 * eps)
            rel = abs(gflat[i] - num) / max(abs(gflat[i]), abs(num), 1e-8)
            pire = max(pire, rel)
            if rel > 1e-4:
                print(f"  ÉCART : {k}[{i}] analytique={gflat[i]:.8f} numérique={num:.8f} rel={rel:.2e}")
    print(f"Plus grand écart relatif : {pire:.2e}")
    print("=> BACKWARD VALIDÉ ✔" if pire < 1e-4 else "=> PROBLÈME ✘ — entraînement interdit par le protocole")
    return pire < 1e-4


def entrainer(args):
    types = [t.strip() for t in args.types.split(",")]
    cfg = {"vocab_size": 0, "d_model": args.d_model, "n_head": args.n_head,
           "types": types, "block_size": args.block_size}
    texte, ids, stoi, itos = charger_donnees(args.data)
    cfg["vocab_size"] = len(itos)
    n_train = int(0.9 * len(ids))
    data_train, data_val = ids[:n_train], ids[n_train:]
    rng = np.random.default_rng(args.seed)
    P = init_params(len(itos), cfg, rng)
    opt = AdamW(P, wd=args.wd)
    os.makedirs(args.out_dir, exist_ok=True)
    historique = [("step", "perte_train", "perte_val")]
    n_par = sum(v.size for v in P.values())
    print(f"[HYBRIDE types={types}] corpus {len(texte):,} car. | V={len(itos)} | params {n_par:,} | seed {args.seed}")
    print(f"{'step':>6} | {'perte train':>11} | {'perte val':>10} | {'lr':>8} | {'tok/s':>7}")
    debut = time.time()
    for step in range(args.max_steps + 1):
        lr = taux_apprentissage(step, args.lr, args.warmup, args.max_steps)
        x, y = prendre_batch(data_train, args.batch_size, cfg["block_size"], rng)
        _, perte, cache = forward(P, cfg, x, y)
        grads = backward(P, cfg, cache, y)
        clip_gradient(grads, args.clip)
        opt.step(grads, lr)
        if step % args.log_every == 0 or step == args.max_steps:
            pv = float(np.mean([forward(P, cfg, *prendre_batch(data_val, args.batch_size, cfg["block_size"], rng))[1] for _ in range(10)]))
            tok_s = int(args.batch_size * cfg["block_size"] * (step + 1) / (time.time() - debut))
            print(f"{step:>6} | {perte:>11.4f} | {pv:>10.4f} | {lr:>8.2e} | {tok_s:>7}")
            historique.append((step, float(perte), pv))
            np.savez_compressed(os.path.join(args.out_dir, "checkpoint.npz"), **P)
            with open(os.path.join(args.out_dir, "historique.csv"), "w", encoding="utf-8") as f:
                for row in historique:
                    f.write(",".join(str(v) for v in row) + "\n")
    vals = [r[2] for r in historique[1:]]
    trs = [r[1] for r in historique[1:]]
    resume = {"types": types, "seed": args.seed, "params": n_par,
              "val_min": min(vals), "val_min_step": historique[1:][int(np.argmin(vals))][0],
              "val_final": vals[-1], "train_final": trs[-1],
              "kv_cache_octets_par_token": sum(2 * cfg["d_model"] * 4 for t in types if t == "smax")}
    with open(os.path.join(args.out_dir, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=1, ensure_ascii=False)
    print("RÉSUMÉ :", json.dumps(resume, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="C1 : hybride linéaire/softmax")
    ap.add_argument("--check-grads", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--types", default="lin,smax", help='ex: "lin,lin", "lin,smax", "smax,smax"')
    ap.add_argument("--data", default="data/corpus_fr.txt")
    ap.add_argument("--out-dir", default="results/c1")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.check_grads:
        ok = verifier_gradients(args)
        raise SystemExit(0 if ok else 1)
    elif args.train:
        entrainer(args)
    else:
        print("Modes : --check-grads (obligatoire) puis --train")

if __name__ == "__main__":
    main()
