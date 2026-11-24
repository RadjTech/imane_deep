#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C7 — vérification indépendante (deuxième script, voie de calcul différente).

1. Intégrité : 25 runs complets (resume.json, historique.csv 19 lignes,
   checkpoint.npz, ENV.json même commit + code_modifie false), cohérence
   resume ↔ historique.
2. Recalcul indépendant des moyennes, différences appariées et IC (quantile de
   Student recalculé par intégration numérique de la loi t, PAS par la table),
   puis comparaison avec results/c7a_analyse.json → IDENTIQUE / ÉCART.
"""
import glob
import json
import math
import os

import numpy as np

CONFIGS = ["smax", "lin-hist", "dlind-hist", "dlind-elu1", "dlind-const"]
SEEDS = [101, 202, 303, 404, 505]
CYCLE = "c7a"


def quantile_t(p, ddl):
    """Quantile t de Student par inversion numérique de la CDF (bisection),
    CDF calculée par intégration de la densité — indépendant de toute table."""
    x = (ddl + 1.0) / 2.0
    const = math.gamma(x) / (math.sqrt(ddl * math.pi) * math.gamma(ddl / 2.0))

    def cdf(t):
        n = 4000
        pas = t / n
        s = 0.0
        for i in range(n):
            u = (i + 0.5) * pas
            s += (1.0 + u * u / ddl) ** (-(ddl + 1) / 2.0) * pas
        return const * s

    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    # ---- 1. intégrité ----
    problemes = []
    commits = set()
    for c in CONFIGS:
        for s in SEEDS:
            d = f"results/{CYCLE}_{c}_s{s}"
            if not os.path.exists(f"{d}/resume.json"):
                problemes.append(f"{d} : resume.json absent"); continue
            if not os.path.exists(f"{d}/checkpoint.npz"):
                problemes.append(f"{d} : checkpoint.npz absent")
            h = open(f"{d}/historique.csv").read().strip().split("\n")
            if len(h) != 19:
                problemes.append(f"{d} : historique {len(h)} lignes (attendu 19)")
            r = json.load(open(f"{d}/resume.json"))
            if abs(r["val_final"] - float(h[-1].split(",")[2])) > 1e-6:
                problemes.append(f"{d} : resume.val_final ≠ historique")
            e = json.load(open(f"{d}/ENV.json"))
            commits.add(e.get("commit", "?"))
            if e.get("code_modifie"):
                problemes.append(f"{d} : code modifié !")
            if not os.path.exists(f"{d}/test.json"):
                problemes.append(f"{d} : test.json absent")
    print(f"intégrité : {'25/25 ✔' if not problemes else 'PROBLÈMES'} | commits vus : {commits or '∅'}")
    for p in problemes:
        print("  -", p)

    # ---- 2. recalcul indépendant ----
    test = {c: {s: json.load(open(f"results/{CYCLE}_{c}_s{s}/test.json"))["test"]
                for s in SEEDS} for c in CONFIGS}

    def ic(diffs):
        x = np.array(diffs, dtype=np.float64)
        n = len(x)
        moy = float(x.mean())
        sd = float(x.std(ddof=1))
        t = quantile_t(0.975, n - 1)
        demi = t * sd / math.sqrt(n)
        return moy, moy - demi, moy + demi

    recal = {}
    for nom, (a, b) in {"C7-V1": ("dlind-elu1", "smax"),
                        "C7-V2": ("dlind-const", "dlind-elu1"),
                        "C7-V3": ("dlind-elu1", "dlind-hist")}.items():
        diffs = [test[a][s] - test[b][s] for s in SEEDS]
        recal[nom] = ic(diffs)
        print(f"{nom} recalculé : {recal[nom][0]:+.4f} IC [{recal[nom][1]:+.4f} ; {recal[nom][2]:+.4f}]")

    # ---- 3. comparaison avec analyse_c7 ----
    ref = json.load(open(f"results/{CYCLE}_analyse.json"))["verdicts"]
    ecart_max = 0.0
    for nom in ("C7-V1", "C7-V2", "C7-V3"):
        r = ref[nom]
        ecart_max = max(ecart_max,
                        abs(r["diff"] - recal[nom][0]),
                        abs(r["ic"][0] - recal[nom][1]),
                        abs(r["ic"][1] - recal[nom][2]))
    print(f"écart max analyse↔vérification : {ecart_max:.2e} → "
          + ("IDENTIQUE ✔" if ecart_max < 5e-4 else "ÉCART — à investiguer"))


if __name__ == "__main__":
    main()
