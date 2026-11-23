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

# Feature map φ de l'attention linéaire et de la delta rule. Deux versions :
#   "elu1"       : ELU + 1 (Katharopoulos et al. 2020) — x+1 si x > 0, exp(x) sinon. Défaut.
#   "historique" : version utilisée en C1–C3 — exp(x)+1 pour x ≤ 0 au lieu de exp(x). Elle saute de
#                  2 à 1 en zéro et vaut au moins 1 partout. Conservée UNIQUEMENT pour reproduire les
#                  chiffres de C1–C3 (--phi historique) ; voir RESEARCH_LOG.md, amendement C4.
# d_phi est commune : la dérivée par morceaux est la même (les différences finies ne voient pas le saut).
PHI = "elu1"

def phi(x):
    if PHI == "historique":
        return np.where(x > 0, x, np.exp(np.minimum(x, 0.0))) + 1.0
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))

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


# ---------- normalisation des clés (stabilité de la delta rule) ----------

def norm_keys_fw(K):
    """k̂ = k/‖k‖₂ par tête : garantit β·‖k̂‖² = β ≤ 1 (contraction) — sans ça
    l'état de la delta rule diverge (constaté au garde-fou : gradients 1e10)."""
    n = np.linalg.norm(K, axis=-1, keepdims=True) + 1e-8
    return K / n, n


def norm_keys_bw(dKhat, K, n):
    """Backward de k̂ = k/n : dk = (dK̂ − k̂·(dK̂·k̂)) / n."""
    Khat = K / n
    dot = (dKhat * Khat).sum(-1, keepdims=True)
    return (dKhat - Khat * dot) / n


# ---------- attention delta rule (par tête) : forme séquentielle + BPTT ----------

def att_delta_fw(Q, K, V, beta):
    """Delta rule — identique à `lin` SAUF la mise à jour de l'état :
        y_t = Q_t·S_{t-1} / (Q_t·Z_{t-1} + ε)
        e_t = V_t − S_{t-1}·k_t                      (trace déjà stockée)
        S_t = S_{t-1} + β·k_t ⊗ e_t                  (remplace au lieu d'additionner)
        Z_t = Z_{t-1} + k_t
    Q,K : sorties de φ (B,H,T,dh) ; V brut ; beta : (H,) dans (0,1).
    Boucle séquentielle sur T (la forme parallèle chunked est hors périmètre C3) ;
    cache complet pour le BPTT."""
    B, H, T, dh = Q.shape
    S = np.zeros((B, H, dh, dh), dtype=Q.dtype)
    Z = np.zeros((B, H, dh), dtype=Q.dtype)
    Y = np.empty_like(Q)
    c = {"Q": Q, "K": K, "V": V, "beta": beta,
         "S_prev": [], "Z_prev": [], "e": [], "num": [], "den": []}
    for t in range(T):
        q, k, v = Q[:, :, t], K[:, :, t], V[:, :, t]          # (B,H,dh)
        num = np.einsum("bhi,bhij->bhj", q, S)
        den = np.einsum("bhi,bhi->bh", q, Z) + EPS_DEN
        y = num / den[..., None]
        e = v - np.einsum("bhij,bhi->bhj", S, k)              # S·k : la trace actuelle
        c["S_prev"].append(S); c["Z_prev"].append(Z)
        c["e"].append(e); c["num"].append(num); c["den"].append(den)
        S = S + beta[None, :, None, None] * k[..., None] * e[:, :, None, :]
        Z = Z + k
        Y[:, :, t] = y
    return Y, c


def att_delta_bw(dY, c):
    """BPTT de la delta rule. Renvoie dQ (après φ), dK (après φ, avant d_phi),
    dV, dβ. Récurrence inverse : dS_{t-1} = dS_t − k⊗de + q⊗dnum."""
    Q, K, V, beta = c["Q"], c["K"], c["V"], c["beta"]
    B, H, T, dh = Q.shape
    dS = np.zeros((B, H, dh, dh), dtype=Q.dtype)   # d/dS_t ; S_T ne feed rien → 0
    dZ_all = np.empty_like(K)
    dQ, dK, dV = np.empty_like(Q), np.zeros_like(K), np.empty_like(V)
    dbeta = np.zeros(H, dtype=Q.dtype)
    for t in reversed(range(T)):
        q, k = Q[:, :, t], K[:, :, t]
        S_prev, Z_prev = c["S_prev"][t], c["Z_prev"][t]
        e, num, den = c["e"][t], c["num"][t], c["den"][t]
        dy = dY[:, :, t]
        # 1) chemin de sortie y = q·S_prev / (q·Z_prev + ε)
        dnum = dy / den[..., None]
        dden = -(dy * num).sum(-1) / (den ** 2)
        dQ[:, :, t] = np.einsum("bhj,bhij->bhi", dnum, S_prev) + dden[..., None] * Z_prev
        dZ_all[:, :, t] = q * dden[..., None]
        # 2) chemin d'erreur — CRUCIAL : dS contient encore le gradient PUR du
        #    futur (∂L/∂S_t). Le terme q⊗dnum (qui alimente S_{t-1} via la
        #    sortie y_t) doit être ajouté APRÈS le calcul de de, sinon il
        #    pollue de, dV et dK.
        de = beta[..., None] * np.einsum("bhij,bhi->bhj", dS, k)   # de_total (∂L/∂e_t)
        dK[:, :, t] += beta[None, :, None] * np.einsum("bhij,bhj->bhi", dS, e)
        dbeta += np.einsum("bhij,bhi,bhj->h", dS, k, e)
        dV[:, :, t] = de
        dK[:, :, t] += -np.einsum("bhij,bhj->bhi", S_prev, de)   # le β est déjà dans de
        # 3) récurrence vers S_{t-1} : pass-through − k⊗de_total (le β est déjà
        #    dans de), PUIS + q⊗dnum
        dS = dS - k[..., None] * de[:, :, None, :]
        dS = dS + np.einsum("bhi,bhj->bhij", q, dnum)
    # chemin Z : dZ_all[t] = gradient wrt Z_{t-1} (état exclusif) — k_s alimente les
    # états Z_j pour j ≥ s, utilisés aux pas t = j+1 → décalage d'un pas (même
    # convention que lin) puis cumsum inverse.
    dZ_shift = np.zeros_like(dZ_all)
    dZ_shift[:, :, :-1] = dZ_all[:, :, 1:]
    dK += revcumsum(dZ_shift, axis=2)
    return dQ, dK, dV, dbeta


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
    beta = None
    if typ == "lin":
        y, att_c = att_lin_fw(q, k, v)
    elif typ == "dlind":
        beta = 1.0 / (1.0 + np.exp(-P[f"b{l}.gamma"].astype(x.dtype)))   # β = sigmoid(γ) ∈ (0,1)
        Kf = phi(k)
        Khat, kn = norm_keys_fw(Kf)
        y, att_c = att_delta_fw(phi(q), Khat, v, beta)
        c["beta"] = beta; c["Kf"] = Kf; c["kn"] = kn
    else:
        y, att_c = att_softmax_fw(q, k, v, scale, mask)
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
    elif typ == "dlind":
        dq, dkhat, dv, dbeta = att_delta_bw(dyh, c["att"])
        dq = dq * d_phi(c["q"])
        dk = norm_keys_bw(dkhat, c["Kf"], c["kn"]) * d_phi(c["k"])
        grads[f"b{l}.gamma"] += dbeta * c["beta"] * (1.0 - c["beta"])   # dβ/dγ
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


# ---------- ternaire (BitNet b1.58) — C4 ----------

CLEFS_MATMUL = ("qkv.W", "proj.W", "fc1.W", "fc2.W")   # poids des matmuls ternarisés par bloc

def quant_ternary(W):
    """Quantification ternaire absmean (BitNet b1.58) :
        γ = mean(|W|)  (échelle, une par tenseur)
        Q = round(clip(W/γ, −1, 1)) ∈ {−1, 0, +1}
        W_eff = γ·Q
    Le backward traverse le quantificateur (STE : dW_eff est appliqué au W maître tel quel).
    Le FD ne peut PAS valider ce transfert (le quantificateur est localement constant autour
    de ses points de sortie) — il est correct par construction ; ce qui EST validé
    numériquement : le backward par rapport à W_eff (verifier_ternaire)."""
    gamma = float(np.mean(np.abs(W)))
    if gamma < 1e-12:
        gamma = 1e-12
    Q = np.clip(np.round(W / gamma), -1.0, 1.0)
    return (gamma * Q).astype(W.dtype), gamma

def parametres_effectifs(P, cfg):
    """P maître (float) → P effectif : les poids des matmuls sont remplacés par γ·Q ;
    embeddings, biais, normalisations et γ restent en float (littérature BitNet :
    couches d'entrée/sortie en haute précision). Le forward/backward float validé
    est réutilisé tel quel sur P_eff — aucune nouvelle voie de gradient."""
    Pe = dict(P)
    for l in range(len(cfg["types"])):
        for s in CLEFS_MATMUL:
            k = f"b{l}.{s}"
            Pe[k], _ = quant_ternary(P[k])
    return Pe


# ---------- modèle ----------

def init_params(V, cfg, rng, dtype=np.float32):
    P, L, D = {}, len(cfg["types"]), cfg["d_model"]
    std, rstd = 0.02, 1.0 / math.sqrt(2.0 * max(1, L))
    P["tok_emb"] = (rng.normal(0, std, (V, D))).astype(dtype)
    P["pos_emb"] = (rng.normal(0, std, (cfg["block_size"], D))).astype(dtype)
    for l in range(L):
        typ = cfg["types"][l]
        if True:  # mêmes paramètres pour tous les types ; γ seulement pour dlind
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
            if typ == "dlind":
                P[f"b{l}.gamma"] = np.zeros(cfg["n_head"], dtype)   # β = sigmoid(γ), init 0.5
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
    ok_global = True
    for types_test in (["lin", "smax"], ["dlind", "dlind"]):
        print(f"[GARDE-FOU] vérification des gradients (float64, types={types_test})...")
        texte, ids, _, itos = charger_donnees(args.data)
        ids = ids[:300]
        cfg = {"vocab_size": len(itos), "d_model": 16, "n_head": 2,
               "types": types_test, "block_size": 12}
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
        print(f"Plus grand écart relatif ({'+'.join(types_test)}) : {pire:.2e}")
        ok = pire < 1e-4
        ok_global = ok_global and ok
    print("=> BACKWARD VALIDÉ ✔" if ok_global else "=> PROBLÈME ✘ — entraînement interdit par le protocole")
    return ok_global


def evaluer_fixe(P, cfg, data_val, T):
    """Évaluation déterministe sur TOUTE la val : fenêtres non chevauchantes de T (B=1), plus une
    dernière fenêtre plus courte, donc chaque caractère est prédit une seule fois (moyenne pondérée).
    Contrairement à l'éval d'entraînement (10 batchs tirés), reproductible post-hoc à l'identique et
    sans tirage aléatoire — utilisée pour choisir checkpoint_best et pour les deux bras de C4."""
    somme, n, i, L = 0.0, 0, 0, len(data_val)
    while i < L - 1:
        r = min(T, L - 1 - i)
        x = data_val[i:i + r][None, :]
        y = data_val[i + 1:i + r + 1][None, :]
        somme += float(forward(P, cfg, x, y)[1]) * r
        n += r
        i += r
    return somme / n


def verifier_phi(chemin_ckpt):
    """Refuse d'évaluer un checkpoint avec une autre φ que celle de son entraînement (resume.json du run)."""
    rj = os.path.join(os.path.dirname(chemin_ckpt), "resume.json")
    if os.path.exists(rj):
        p = json.load(open(rj, encoding="utf-8")).get("phi")
        if p and p != PHI:
            raise SystemExit(f"{chemin_ckpt} : entraîné avec φ={p}, évalué avec φ={PHI} — préciser --phi {p}")


def verifier_ternaire(args):
    """Garde-fou C4 : (a) tests unitaires du quantificateur ; (b) FD du backward wrt
    P_eff (les poids effectifs — continus, donc FD valide) sur config dlind,dlind ;
    (c) note honnête : le transfert STE maître←dW_eff est par construction, non FD-ifiable."""
    ok = True
    # (a) quantificateur
    rng = np.random.default_rng(2)
    for forme in ((64, 32), (10,), (5, 1)):
        W = rng.normal(0, 0.1, forme)
        W_eff, g = quant_ternary(W)
        Q = np.unique(W_eff / g)
        bon = set(Q) <= {-1.0, 0.0, 1.0} and g > 0 and np.abs(W_eff - g * np.clip(np.round(W / g), -1, 1)).max() == 0
        ok &= bon
        print(f"  quantificateur {forme} : valeurs {sorted(Q)} | γ={g:.4f} | {'✔' if bon else '✘'}")
    W0 = np.zeros((4, 4)); W0e, g0 = quant_ternary(W0)
    ok &= bool(np.all(W0e == 0) and g0 > 0)
    print(f"  quantificateur tenseur nul : W_eff=0, γ>0 {'✔' if ok else '✘'} (garde)")
    # (b) FD sur P_eff (dlind,dlind)
    types_test = ["dlind", "dlind"]
    texte, ids, _, itos = charger_donnees(args.data)
    ids = ids[:300]
    cfg = {"vocab_size": len(itos), "d_model": 16, "n_head": 2,
           "types": types_test, "block_size": 12}
    P = init_params(len(itos), cfg, np.random.default_rng(0), dtype=np.float64)
    Pe = parametres_effectifs(P, cfg)
    # vérifier que Pe est réellement ternaire sur les clés matmul
    for l in range(len(types_test)):
        for s in CLEFS_MATMUL:
            k = f"b{l}.{s}"
            _, g = quant_ternary(P[k])
            ok &= bool(set(np.unique(Pe[k] / g)) <= {-1.0, 0.0, 1.0})
    x, y = prendre_batch(ids, B=2, T=cfg["block_size"], rng=np.random.default_rng(1))
    _, perte, cache = forward(Pe, cfg, x, y)
    grads = backward(Pe, cfg, cache, y)
    rng2 = np.random.default_rng(1)
    eps = 1e-5
    pire = 0.0
    for k in Pe:
        flat, gflat = Pe[k].reshape(-1), grads[k].reshape(-1)
        for i in rng2.integers(0, flat.size, size=min(4, flat.size)):
            sauve = flat[i]
            flat[i] = sauve + eps; lp = forward(Pe, cfg, x, y)[1]
            flat[i] = sauve - eps; lm = forward(Pe, cfg, x, y)[1]
            flat[i] = sauve
            num = (lp - lm) / (2 * eps)
            # plancher absolu pour le bruit d'annulation du FD : si |g−num| < 1e-8, les deux
            # sont d'accord (un vrai bug donne |g−num| ~ |g| lui-même, soit ≫ ce plancher)
            d_abs = abs(gflat[i] - num)
            rel = d_abs / max(abs(gflat[i]), abs(num), 1e-8)
            if d_abs >= 1e-8:
                pire = max(pire, rel)
            if rel > 1e-4 and d_abs >= 1e-8:
                print(f"  ÉCART : {k}[{i}] analytique={gflat[i]:.8f} numérique={num:.8f} rel={rel:.2e} d_abs={d_abs:.2e}")
    print(f"Plus grand écart relatif FD sur P_eff (dlind,dlind) : {pire:.2e} (plancher absolu 1e-8)")
    ok &= pire < 1e-4
    print("(c) transfert STE (dW_eff → W maître) : par construction, non vérifiable par FD — consigné")
    print("=> TERNAIRE VALIDÉ ✔" if ok else "=> PROBLÈME ✘ — entraînement ternaire interdit par le protocole")
    return ok


def quantifier_apres(args):
    """Bras B de H3 : charger un checkpoint float, ternariser SANS ré-entraînement, évaluer avec
    l'eval fixe. Δ = val_ternaire − val_float sur le MÊME eval déterministe, à effectuer au MÊME
    point que le bras A (checkpoint_best ou checkpoint final)."""
    verifier_phi(args.quantize_after)
    P = dict(np.load(args.quantize_after))
    types = [t.strip() for t in args.types.split(",")]
    texte, ids, stoi, itos = charger_donnees(args.data)
    cfg = {"vocab_size": len(itos), "d_model": args.d_model, "n_head": args.n_head,
           "types": types, "block_size": args.block_size}
    data_val = ids[int(0.9 * len(ids)):]
    vf = evaluer_fixe(P, cfg, data_val, cfg["block_size"])
    Pe = parametres_effectifs(P, cfg)
    vt = evaluer_fixe(Pe, cfg, data_val, cfg["block_size"])
    etiquette = "best" if "best" in os.path.basename(args.quantize_after) else "final"
    resume = {"ckpt": args.quantize_after, "point": etiquette, "eval": "fenêtres fixes, val entière",
              "phi": PHI, "val_float_fixe": round(vf, 6), "val_ternaire_fixe": round(vt, 6),
              "delta_ternaire": round(vt - vf, 6)}
    os.makedirs(args.out_dir, exist_ok=True)
    seed = args.seed
    chemin = os.path.join(args.out_dir, f"quantize_after_{etiquette}_s{seed}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=1, ensure_ascii=False)
    print("RÉSUMÉ :", json.dumps(resume, ensure_ascii=False))


def exporter_ternaire(args):
    """Exporte P_eff (poids effectifs γ·Q en float32 + échelles) pour le profileur."""
    verifier_phi(args.export_ternary)
    P = dict(np.load(args.export_ternary))
    types = [t.strip() for t in args.types.split(",")]
    cfg = {"vocab_size": P["tok_emb"].shape[0], "d_model": args.d_model, "n_head": args.n_head,
           "types": types, "block_size": args.block_size}
    Pe = parametres_effectifs(P, cfg)
    echelles = {}
    for l in range(len(types)):
        for s in CLEFS_MATMUL:
            _, g = quant_ternary(P[f"b{l}.{s}"])
            echelles[f"b{l}.{s[:-2]}scale"] = np.float32(g)
    os.makedirs(os.path.dirname(args.out_ternary) or ".", exist_ok=True)
    np.savez_compressed(args.out_ternary, **Pe, **echelles)
    print(f"Checkpoint ternaire exporté : {args.out_ternary} ({len(echelles)} échelles)")


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
    historique = [("step", "perte_train", "perte_val", "perte_val_fixe")]
    n_par = sum(v.size for v in P.values())
    prefix = " [TERNAIRE matmuls : γ·Q, STE]" if args.ternary else ""
    print(f"[HYBRIDE types={types}]{prefix} φ={PHI} corpus {len(texte):,} car. | V={len(itos)} | params {n_par:,} | seed {args.seed}")
    print(f"{'step':>6} | {'perte train':>11} | {'perte val':>10} | {'val fixe':>9} | {'lr':>8} | {'tok/s':>7}")
    meilleure = (float("inf"), -1)          # (val fixe, pas) du checkpoint_best
    debut = time.time()
    for step in range(args.max_steps + 1):
        lr = taux_apprentissage(step, args.lr, args.warmup, args.max_steps)
        x, y = prendre_batch(data_train, args.batch_size, cfg["block_size"], rng)
        Pe = parametres_effectifs(P, cfg) if args.ternary else P
        _, perte, cache = forward(Pe, cfg, x, y)
        grads = backward(Pe, cfg, cache, y)          # grads quantifiés = STE (dW_eff → maître)
        clip_gradient(grads, args.clip)
        opt.step(grads, lr)
        if step % args.log_every == 0 or step == args.max_steps:
            Pev = parametres_effectifs(P, cfg) if args.ternary else P
            pv = float(np.mean([forward(Pev, cfg, *prendre_batch(data_val, args.batch_size, cfg["block_size"], rng))[1] for _ in range(10)]))
            pf = evaluer_fixe(Pev, cfg, data_val, cfg["block_size"])   # déterministe, aucun tirage rng
            tok_s = int(args.batch_size * cfg["block_size"] * (step + 1) / (time.time() - debut))
            print(f"{step:>6} | {perte:>11.4f} | {pv:>10.4f} | {pf:>9.4f} | {lr:>8.2e} | {tok_s:>7}")
            historique.append((step, float(perte), pv, pf))
            np.savez_compressed(os.path.join(args.out_dir, "checkpoint.npz"), **P)
            if pf < meilleure[0]:
                meilleure = (pf, step)
                np.savez_compressed(os.path.join(args.out_dir, "checkpoint_best.npz"), **P)
            with open(os.path.join(args.out_dir, "historique.csv"), "w", encoding="utf-8") as f:
                for row in historique:
                    f.write(",".join(str(v) for v in row) + "\n")
    vals = [r[2] for r in historique[1:]]
    trs = [r[1] for r in historique[1:]]
    fixes = [r[3] for r in historique[1:]]
    resume = {"types": types, "seed": args.seed, "params": n_par,
              "ternary": bool(args.ternary), "phi": PHI,
              "val_min": min(vals), "val_min_step": historique[1:][int(np.argmin(vals))][0],
              "val_final": vals[-1], "train_final": trs[-1],
              "val_fixe_min": meilleure[0], "val_fixe_min_step": meilleure[1],
              "val_fixe_final": fixes[-1],
              "kv_cache_octets_par_token": sum(2 * cfg["d_model"] * 4 for t in types if t == "smax")}
    with open(os.path.join(args.out_dir, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=1, ensure_ascii=False)
    print("RÉSUMÉ :", json.dumps(resume, ensure_ascii=False))


def main():
    global PHI
    ap = argparse.ArgumentParser(description="C1 : hybride linéaire/softmax")
    ap.add_argument("--phi", choices=["elu1", "historique"], default="elu1",
                    help="feature map : ELU+1 (défaut) ; « historique » = version boguée de C1–C3, pour reproduction")
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
    ap.add_argument("--ternary", action="store_true",
                    help="entraînement avec poids des matmuls ternarisés (BitNet b1.58, STE)")
    ap.add_argument("--check-ternary", action="store_true",
                    help="garde-fou C4 : quantificateur + FD wrt poids effectifs")
    ap.add_argument("--quantize-after", default=None, metavar="CKPT",
                    help="bras B : ternariser un checkpoint float sans ré-entraînement, évaluer")
    ap.add_argument("--export-ternary", default=None, metavar="CKPT",
                    help="exporter P_eff (γ·Q + échelles) pour le profileur")
    ap.add_argument("--out-ternary", default="results/c4/checkpoint_ternaire.npz")
    args = ap.parse_args()
    PHI = args.phi
    if args.check_grads:
        ok = verifier_gradients(args)
        raise SystemExit(0 if ok else 1)
    elif args.check_ternary:
        ok = verifier_ternaire(args)
        raise SystemExit(0 if ok else 1)
    elif args.quantize_after:
        quantifier_apres(args)
    elif args.export_ternary:
        exporter_ternaire(args)
    elif args.train:
        entrainer(args)
    else:
        print("Modes : --check-grads puis --check-ternary (obligatoires) puis --train [--ternary]")

if __name__ == "__main__":
    main()
