#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cycle C3b — contrôles : φ corrigée et ablation réellement propre de C3.

Importe hybrid_numpy SANS le modifier et remplace φ ou l'attention selon --variante :

  smax         smax,smax (code du dépôt)
  hybride-hist lin,smax avec la φ du dépôt         (= C1/C2)
  hybride-elu1 lin,smax avec φ = ELU+1
  lin-hist     lin,lin avec la φ du dépôt
  lin-elu1     lin,lin avec φ = ELU+1
  lin-hist-nk  lin-hist + clés normalisées       (isole « normalisation » de « delta »)
  lin-elu1-nk  lin-elu1 + clés normalisées
  dlind-hist   dlind,dlind avec la φ du dépôt    (= C3)
  dlind-elu1   dlind,dlind avec φ = ELU+1
  dlind-const  dlind avec clés CONSTANTES : oubli exponentiel pur, sans adressage par contenu
  dlind-noz    dlind-elu1 sans dénominateur Z (requêtes normalisées)

Mêmes tirages aléatoires et même protocole que hybrid_numpy.entrainer ; s'y ajoute une
évaluation déterministe de la val ENTIÈRE (colonne perte_val_entiere).

  python3 c3b_controles.py --check
  python3 c3b_controles.py --variante lin-elu1 --seed 1337 --out-dir results/c3b_lin-elu1_s1337
"""
import argparse
import json
import math
import os
import sys
import time


def racine_depot():
    """Remonte jusqu'au dossier qui contient experiments/hybrid_numpy.py."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


RACINE = os.environ.get("IMANE_RACINE") or racine_depot()
sys.path.insert(0, os.path.join(RACINE, "experiments"))
import numpy as np  # noqa: E402
import hybrid_numpy as hn  # noqa: E402

REF = {n: getattr(hn, n) for n in ("phi", "gelu", "d_gelu", "att_lin_fw", "att_lin_bw", "att_delta_fw", "att_delta_bw")}
#             types par couche   φ       clés norm.  forme delta
VARIANTES = {
    "smax":        ("smax,smax",   "hist", False, "ref"),
    "hybride-hist": ("lin,smax",   "hist", False, "ref"),
    "hybride-elu1": ("lin,smax",   "elu1", False, "ref"),
    "lin-hist":    ("lin,lin",     "hist", False, "ref"),
    "lin-elu1":    ("lin,lin",     "elu1", False, "ref"),
    "lin-hist-nk": ("lin,lin",     "hist", True,  "ref"),
    "lin-elu1-nk": ("lin,lin",     "elu1", True,  "ref"),
    "dlind-hist":  ("dlind,dlind", "hist", False, "ref"),
    "dlind-elu1":  ("dlind,dlind", "elu1", False, "ref"),
    "dlind-const": ("dlind,dlind", "elu1", False, "const"),
    "dlind-noz":   ("dlind,dlind", "elu1", False, "noz"),
}
ETAT = {"normkeys": False, "delta": "ref"}


def phi_elu1(x):
    """ELU + 1 (Katharopoulos et al. 2020) : x+1 si x > 0, exp(x) sinon. hn.d_phi reste valable."""
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0))).astype(x.dtype)


# gelu : mêmes formules, x*x*x au lieu de x**3 (vitesse seulement)
_c = math.sqrt(2.0 / math.pi)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(_c * (x + 0.044715 * x * x * x)))


def d_gelu(x):
    t = np.tanh(_c * (x + 0.044715 * x * x * x))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (_c * (1.0 + 3.0 * 0.044715 * x * x))


# ---------- lin : forme noyau T×T, égale à la forme état ----------

def lin_fw(q, k, v):
    Q, Kb = hn.phi(q), hn.phi(k)
    K, n = hn.norm_keys_fw(Kb) if ETAT["normkeys"] else (Kb, None)
    L = np.tril(np.ones((Q.shape[2],) * 2, Q.dtype), -1)
    A = (Q @ K.transpose(0, 1, 3, 2)) * L
    num, den = A @ v, A.sum(-1) + hn.EPS_DEN
    return num / den[..., None], {"Q": Q, "K": K, "Kb": Kb, "n": n, "v": v, "A": A, "L": L, "num": num, "den": den}


def lin_bw(dy, c):
    dnum = dy / c["den"][..., None]
    dden = -(dy * c["num"]).sum(-1) / c["den"] ** 2
    dA = (dnum @ c["v"].transpose(0, 1, 3, 2) + dden[..., None]) * c["L"]
    dK = dA.transpose(0, 1, 3, 2) @ c["Q"]
    if c["n"] is not None:
        dK = hn.norm_keys_bw(dK, c["Kb"], c["n"])
    return dA @ c["K"], dK, c["A"].transpose(0, 1, 3, 2) @ dnum


# ---------- delta : forme triangulaire exacte, variantes const / noz ----------

def delta_fw(Q, K, V, beta):
    T = Q.shape[2]
    if ETAT["delta"] == "const":                  # K̂ constant ⇒ y_t = (1/t)·Σ_{s<t} β(1−β)^{t−1−s} v_s
        t = np.arange(T)
        n = t[:, None] - 1 - t[None, :]
        m, nn, b = n >= 0, np.maximum(n, 0), beta[:, None, None]
        pw = np.where(m, (1 - b) ** nn, 0.0)
        pm = np.where(m & (nn > 0), (1 - b) ** np.maximum(nn - 1, 0), 0.0)
        tt = np.maximum(t, 1)[None, :, None]
        A, dA = (b * pw / tt).astype(V.dtype), ((pw - b * nn * pm) / tt).astype(V.dtype)
        return A[None] @ V, {"Q": Q, "K": K, "V": V, "A": A, "dA": dA}
    L1 = np.tril(np.ones((T, T), Q.dtype), -1)
    noz = ETAT["delta"] == "noz"
    Qn, n = hn.norm_keys_fw(Q) if noz else (Q, None)
    b = beta[None, :, None, None].astype(Q.dtype)
    L = (K @ K.transpose(0, 1, 3, 2)) * L1
    Aq = (Qn @ K.transpose(0, 1, 3, 2)) * L1
    M = np.eye(T, dtype=Q.dtype) + b * L
    E = np.linalg.solve(M, V)                     # (I + β·tril(K̂K̂ᵀ,−1)) E = V
    num = b * (Aq @ E)
    den = None if noz else Aq.sum(-1) + hn.EPS_DEN
    y = num if noz else num / den[..., None]
    return y, {"Q": Q, "Qn": Qn, "n": n, "K": K, "V": V, "b": b, "L": L, "L1": L1, "Aq": Aq, "M": M, "E": E,
               "num": num, "den": den}


def delta_bw(dY, c):
    if "dA" in c:
        dbeta = np.einsum("hts,bhtj,bhsj->h", c["dA"], dY, c["V"], optimize=True)
        return np.zeros_like(c["Q"]), np.zeros_like(c["K"]), c["A"].transpose(0, 2, 1)[None] @ dY, dbeta
    K, b, L, L1, Aq, E, den = c["K"], c["b"], c["L"], c["L1"], c["Aq"], c["E"], c["den"]
    dnum = dY if den is None else dY / den[..., None]
    ET = E.transpose(0, 1, 3, 2)
    dAq = b * (dnum @ ET)
    if den is not None:
        dAq = dAq - ((dY * c["num"]).sum(-1) / den ** 2)[..., None]
    dAq = dAq * L1
    lam = np.linalg.solve(c["M"].transpose(0, 1, 3, 2), b * (Aq.transpose(0, 1, 3, 2) @ dnum))
    dM = -(lam @ ET)
    dL = (b * dM) * L1
    dQ = dAq @ K
    dK = dAq.transpose(0, 1, 3, 2) @ c["Qn"] + (dL + dL.transpose(0, 1, 3, 2)) @ K
    dbeta = (dM * L).sum((0, 2, 3)) + (dnum * (Aq @ E)).sum((0, 2, 3))
    if c["n"] is not None:
        dQ = hn.norm_keys_bw(dQ, c["Q"], c["n"])
    return dQ, dK, lam, dbeta


def installer(phi, normkeys, delta):
    ETAT.update(normkeys=normkeys, delta=delta)
    hn.phi = phi_elu1 if phi == "elu1" else REF["phi"]
    hn.gelu, hn.d_gelu = gelu, d_gelu
    hn.att_lin_fw, hn.att_lin_bw = lin_fw, lin_bw
    hn.att_delta_fw, hn.att_delta_bw = (REF["att_delta_fw"], REF["att_delta_bw"]) if delta == "ref" else (delta_fw, delta_bw)


def desinstaller():
    for n, f in REF.items():
        setattr(hn, n, f)


# ---------- garde-fou ----------

def verifier():
    """Équivalence avec le dépôt (variantes -hist) puis différences finies de TOUTES les variantes."""
    _, ids, _, itos = hn.charger_donnees(os.path.join(RACINE, "data", "corpus_fr.txt"))
    ok = True
    for types, delta in (("lin,lin", "ref"), ("dlind,dlind", "tri")):
        cfg = {"vocab_size": len(itos), "d_model": 96, "n_head": 4, "types": types.split(","), "block_size": 96}
        rng = np.random.default_rng(3)
        P = hn.init_params(len(itos), cfg, rng, dtype=np.float64)
        x, y = hn.prendre_batch(ids[:9000], 2, 96, rng)
        desinstaller()
        _, l0, c0 = hn.forward(P, cfg, x, y)
        g0 = hn.backward(P, cfg, c0, y)
        installer("hist", False, delta)
        _, l1, c1 = hn.forward(P, cfg, x, y)
        g1 = hn.backward(P, cfg, c1, y)
        e = max(float(np.abs(g0[k] - g1[k]).max() / (np.abs(g0[k]).max() + 1e-30)) for k in g0)
        bon = abs(float(l0 - l1)) < 1e-12 and e < 1e-9
        ok &= bon
        print(f"[{'OK   ' if bon else 'ÉCHEC'}] forme rapide = dépôt ({types}) : perte {abs(float(l0 - l1)):.1e}, gradients {e:.1e}")
    for nom, (types, phi, nk, delta) in VARIANTES.items():
        installer(phi, nk, delta)
        cfg = {"vocab_size": len(itos), "d_model": 16, "n_head": 2, "types": types.split(","), "block_size": 12}
        rng = np.random.default_rng(0)
        P = hn.init_params(len(itos), cfg, rng, dtype=np.float64)
        for k in P:
            if k.endswith("qkv.W"):
                P[k] *= 20                         # sort du régime quasi nul de l'init
            if k.endswith("gamma"):
                P[k] += np.array([0.3, -0.4])
        x, y = hn.prendre_batch(ids[:300], 2, 12, rng)
        _, _, c = hn.forward(P, cfg, x, y)
        g, pire, r2 = hn.backward(P, cfg, c, y), 0.0, np.random.default_rng(1)
        for k in P:
            f, gf = P[k].reshape(-1), g[k].reshape(-1)
            for i in r2.integers(0, f.size, size=min(6, f.size)):
                s = f[i]
                f[i] = s + 1e-5; lp = hn.forward(P, cfg, x, y)[1]
                f[i] = s - 1e-5; lm = hn.forward(P, cfg, x, y)[1]
                f[i] = s
                num = (lp - lm) / 2e-5
                if max(abs(num), abs(gf[i])) > 1e-7:
                    pire = max(pire, abs(num - gf[i]) / max(abs(num), abs(gf[i])))
        bon = pire < 1e-4
        ok &= bon
        print(f"[{'OK   ' if bon else 'ÉCHEC'}] différences finies {nom:<12} : pire écart {pire:.1e}")
    return ok


# ---------- entraînement (protocole de hybrid_numpy.entrainer) ----------

def fenetres(val, T):
    d = list(range(0, len(val) - T - 1, 16)) + [len(val) - T - 1]
    return np.stack([val[i:i + T] for i in d]), np.stack([val[i + 1:i + T + 1] for i in d])


def eval_entiere(P, cfg, X, Y, B=16):
    tot = n = 0
    for i in range(0, len(X), B):
        m = X[i:i + B].size
        tot += float(hn.forward(P, cfg, X[i:i + B], Y[i:i + B])[1]) * m
        n += m
    return tot / n


def entrainer(a):
    types, phi, nk, delta = VARIANTES[a.variante]
    installer(phi, nk, delta)
    if os.path.exists(os.path.join(a.out_dir, "resume.json")):
        raise SystemExit(f"{a.out_dir} contient déjà un run terminé (règle 4 : on n'écrase pas)")
    _, ids, _, itos = hn.charger_donnees(a.data)
    cfg = {"vocab_size": len(itos), "d_model": 96, "n_head": 4, "types": types.split(","), "block_size": 96}
    n = int(0.9 * len(ids))
    tr, va = ids[:n], ids[n:]
    Xv, Yv = fenetres(va, 96)
    rng = np.random.default_rng(a.seed)
    P = hn.init_params(len(itos), cfg, rng)
    opt = hn.AdamW(P, wd=0.05)
    os.makedirs(a.out_dir, exist_ok=True)
    h, t0 = [], time.time()
    for pas in range(a.max_steps + 1):
        lr = hn.taux_apprentissage(pas, 1e-3, 100, a.max_steps)
        x, y = hn.prendre_batch(tr, 16, 96, rng)
        _, perte, cache = hn.forward(P, cfg, x, y)
        g = hn.backward(P, cfg, cache, y)
        hn.clip_gradient(g, 1.0)
        opt.step(g, lr)
        if pas % 100 == 0 or pas == a.max_steps:
            pv = float(np.mean([hn.forward(P, cfg, *hn.prendre_batch(va, 16, 96, rng))[1] for _ in range(10)]))
            pe = eval_entiere(P, cfg, Xv, Yv)
            h.append((pas, float(perte), pv, pe))
            print(f"{pas:>5} | train {float(perte):.4f} | val {pv:.4f} | val entière {pe:.4f} | {time.time() - t0:.0f}s", flush=True)
            with open(os.path.join(a.out_dir, "historique.csv"), "w", encoding="utf-8") as f:
                f.write("step,perte_train,perte_val,perte_val_entiere\n")
                f.writelines(",".join(map(str, r)) + "\n" for r in h)
    np.savez_compressed(os.path.join(a.out_dir, "checkpoint.npz"), **P)
    vm = [r[2] for r in h]
    res = {"variante": a.variante, "types": types.split(","), "phi": phi, "cles_normalisees": nk, "delta": delta,
           "seed": a.seed, "params": sum(v.size for v in P.values()), "val_min": min(vm),
           "val_min_step": h[int(np.argmin(vm))][0], "val_final": vm[-1], "train_final": h[-1][1],
           "val_entiere_min": min(r[3] for r in h), "val_entiere_final": h[-1][3]}
    json.dump(res, open(os.path.join(a.out_dir, "resume.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RÉSUMÉ :", json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variante", choices=VARIANTES)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--data", default=os.path.join(RACINE, "data", "corpus_fr.txt"))
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if verifier() else 1)
    if not a.variante or not a.out_dir:
        raise SystemExit("--variante et --out-dir sont obligatoires (ou --check)")
    entrainer(a)


if __name__ == "__main__":
    main()
