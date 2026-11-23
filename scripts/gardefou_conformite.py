#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Garde-fou de CONFORMITÉ — complète --check-grads.

Les garde-fous existants vérifient la cohérence interne : le backward est bien la
dérivée du forward, l'incrémental est égal au parallèle, le script unifié est égal
au script du cours. Aucun ne compare le forward à sa SPÉCIFICATION. Une formule
fausse mais dérivable par morceaux passe donc tous les tests (c'est ce qui est
arrivé à phi). Ce script vérifie les formules elles-mêmes.

Usage : python3 scripts/gardefou_conformite.py        (code retour 1 si un test échoue)
"""
import os
import sys

import numpy as np

def racine_depot():
    """Remonte jusqu'au dossier qui contient experiments/hybrid_numpy.py (marche depuis scripts/ ou propositions/scripts/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


sys.path.insert(0, os.path.join(racine_depot(), "experiments"))
import hybrid_numpy as hn  # noqa: E402

echecs = 0


def verdict(nom, ok, detail):
    global echecs
    echecs += 0 if ok else 1
    print(f"[{'OK   ' if ok else 'ÉCHEC'}] {nom} : {detail}")


def modele(types, d=32, H=2, T=24, V=20, seed=0, gain=12.0):
    """Petit modèle float64 sorti du régime quasi nul de l'init (poids qkv × gain)."""
    cfg = {"vocab_size": V, "d_model": d, "n_head": H, "types": types, "block_size": T}
    rng = np.random.default_rng(seed)
    P = hn.init_params(V, cfg, rng, dtype=np.float64)
    for k in P:
        if k.endswith("qkv.W"):
            P[k] = P[k] * gain
    return P, cfg, rng.integers(0, V, size=(2, T))


# 1. φ = ELU + 1 (Katharopoulos et al. 2020) : x+1 si x>0, exp(x) sinon ----------------
x = np.linspace(-6, 6, 4801)
ref = np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))
ecart = float(np.abs(hn.phi(x) - ref).max())
verdict("phi = ELU+1", ecart < 1e-12, f"écart max {ecart:.3g}")
h = 1e-9
saut = float(abs(hn.phi(np.array([h]))[0] - hn.phi(np.array([-h]))[0]))
verdict("phi continue en 0", saut < 1e-6, f"saut {saut:.3g} entre -1e-9 et +1e-9")
verdict("phi croissante", bool((np.diff(hn.phi(x)) >= -1e-12).all()), "attendu : jamais décroissante")
verdict("phi > 0 et tend vers 0", bool(hn.phi(np.array([-30.0]))[0] < 1e-6),
        f"phi(-30) = {hn.phi(np.array([-30.0]))[0]:.3g}")

# 2. Causalité stricte : modifier le futur ne change aucun logit du passé ----------------
for types in (["smax", "smax"], ["lin", "lin"], ["dlind", "dlind"], ["lin", "smax"]):
    P, cfg, idx = modele(types)
    l0 = hn.forward(P, cfg, idx)[0]
    idx2 = idx.copy()
    idx2[:, 13:] = (idx2[:, 13:] + 7) % cfg["vocab_size"]
    l1 = hn.forward(P, cfg, idx2)[0]
    fuite = float(np.abs(l0[:, :13] - l1[:, :13]).max())
    verdict(f"causalité {'+'.join(types)}", fuite < 1e-12, f"fuite max {fuite:.2e}")

# 3. lin : forme état = forme noyau T×T --------------------------------------------------
rng = np.random.default_rng(1)
q, k, v = (rng.normal(size=(2, 3, 20, 8)) for _ in range(3))
y_etat, _ = hn.att_lin_fw(q, k, v)
Q, K = hn.phi(q), hn.phi(k)
A = (Q @ K.transpose(0, 1, 3, 2)) * np.tril(np.ones((20, 20)), -1)
y_noyau = (A @ v) / (A.sum(-1) + hn.EPS_DEN)[..., None]
e = float(np.abs(y_etat - y_noyau).max())
verdict("lin : état = noyau", e < 1e-12, f"écart {e:.2e}")

# 4. dlind : récurrence = système triangulaire (I + β·tril(K̂K̂ᵀ,-1)) E = V ---------------
Kh, _ = hn.norm_keys_fw(K)
beta = np.array([0.3, 0.55, 0.9])
y_rec, _ = hn.att_delta_fw(Q, Kh, v, beta)
Lm = np.tril(np.ones((20, 20)), -1)
b = beta[None, :, None, None]
E = np.linalg.solve(np.eye(20) + b * (Kh @ Kh.transpose(0, 1, 3, 2)) * Lm, v)
Aq = (Q @ Kh.transpose(0, 1, 3, 2)) * Lm
y_tri = b * (Aq @ E) / (Aq.sum(-1) + hn.EPS_DEN)[..., None]
e = float(np.abs(y_rec - y_tri).max())
verdict("dlind : récurrence = système triangulaire", e < 1e-12, f"écart {e:.2e}")

# 5. Gradients : TOUTES les entrées (petites dims), pas 4 par tenseur ---------------------
for types in (["lin", "smax"], ["dlind", "dlind"], ["smax", "smax"]):
    P, cfg, idx = modele(types, d=8, H=2, T=6, V=7, gain=12.0)
    cible = np.roll(idx, -1, axis=1)
    _, _, c = hn.forward(P, cfg, idx, cible)
    g = hn.backward(P, cfg, c, cible)
    pire = (0.0, "")
    for nom in P:
        f, gf = P[nom].reshape(-1), g[nom].reshape(-1)
        num = np.empty_like(gf)
        for i in range(f.size):
            s = f[i]
            f[i] = s + 1e-6; lp = hn.forward(P, cfg, idx, cible)[1]
            f[i] = s - 1e-6; lm = hn.forward(P, cfg, idx, cible)[1]
            f[i] = s
            num[i] = (lp - lm) / 2e-6
        rel = float(np.abs(num - gf).max() / max(np.abs(gf).max(), np.abs(num).max(), 1e-12))
        pire = max(pire, (rel, nom))
    verdict(f"gradients complets {'+'.join(types)}", pire[0] < 1e-5, f"pire écart {pire[0]:.1e} ({pire[1]})")

print(f"\n=> {'CONFORME' if echecs == 0 else f'{echecs} test(s) en échec'}")
sys.exit(1 if echecs else 0)
