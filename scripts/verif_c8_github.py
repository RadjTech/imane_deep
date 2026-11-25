#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C8 (GitHub Actions) — vérification indépendante des verdicts.

Recalcule moyennes, différences appariées par seed et IC 95 % de Student à partir
des test.json de results/c8g_*, avec un quantile de Student recalculé par
intégration numérique (PAS de table, PAS le même code que analyse_c8_github.py),
puis compare à results/c8g_analyse.json (produit par scripts/analyse_c8_github.py).

Critères (PROTOCOLE_C8_GITHUB.md, métrique principale = test_best) :
  C8-V0 : à 1 800 pas, IC(dlind-elu1 − smax) entièrement > 0 (réplication C7-V1).
  C8-V1 : à 18 000 pas, IC(dlind-elu1 − smax) entièrement < 0.
  C8-V2 : à 5 400 pas,  IC(dlind-elu1 − smax) entièrement < 0.
  C8-V3 : à 18 000 pas, borne haute de IC(dlind-const − dlind-elu1) ≤ +0.05.
  C8-V4 : IC de la différence d'écarts (dlind-elu1 − smax)@18000 − @1800 ne contient pas 0.
"""
import glob
import json
import math
import os
import sys

import numpy as np

SEEDS = [611, 622, 633, 644, 655]
BUDGETS = [1800, 5400, 18000]
VARIANTES = ["smax", "dlind-elu1", "dlind-const"]


def quantile_t(p, ddl):
    x = (ddl + 1.0) / 2.0
    const = math.gamma(x) / (math.sqrt(ddl * math.pi) * math.gamma(ddl / 2.0))

    def cdf(t):
        n = 4000
        pas = t / n
        s = 0.0
        for i in range(n):
            u = (i + 0.5) * pas
            s += (1.0 + u * u / ddl) ** (-(ddl + 1) / 2.0) * pas
        return 0.5 + const * s

    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ic(diffs):
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    moy = float(x.mean())
    sd = float(x.std(ddof=1))
    demi = quantile_t(0.975, n - 1) * sd / math.sqrt(n)
    return moy, moy - demi, moy + demi


def charger(metric="test_best"):
    d = {}
    for v in VARIANTES:
        d[v] = {}
        for b in BUDGETS:
            d[v][b] = {}
            for s in SEEDS:
                chemin = f"results/c8g_{v}_b{b}_s{s}/test.json"
                if not os.path.exists(chemin):
                    raise SystemExit(f"il manque {chemin}")
                d[v][b][s] = json.load(open(chemin))[metric]
    return d


def main():
    d = charger("test_best")
    print("=" * 78)
    print("C8 — vérification indépendante (test_best, apparié par seed, quantile recalculé)")
    print("=" * 78)
    for v in VARIANTES:
        for b in BUDGETS:
            vals = np.array([d[v][b][s] for s in SEEDS])
            print(f"{v:12s} b{b:6d} : " + " ".join(f"{x:.4f}" for x in vals)
                  + f" | moy {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

    def diff(v, b, w):
        return [d[v][b][s] - d[w][b][s] for s in SEEDS]

    out = {}
    i0 = ic(diff("dlind-elu1", 1800, "smax"))
    out["C8-V0"] = {"diff": i0[0], "ic": i0[1:],
                    "verdict": "confirmée" if i0[2] > 0 else "infirmée"}
    i54 = ic(diff("dlind-elu1", 5400, "smax"))
    out["C8-V2"] = {"diff": i54[0], "ic": i54[1:],
                    "verdict": "confirmée" if i54[2] < 0 else ("infirmée" if i54[1] > 0 else "indéterminée")}
    i18 = ic(diff("dlind-elu1", 18000, "smax"))
    out["C8-V1"] = {"diff": i18[0], "ic": i18[1:],
                    "verdict": "confirmée" if i18[2] < 0 else ("infirmée" if i18[1] > 0 else "indéterminée")}
    ic3 = ic(diff("dlind-const", 18000, "dlind-elu1"))
    out["C8-V3"] = {"diff": ic3[0], "ic": ic3[1:],
                    "verdict": "confirmée" if ic3[2] <= 0.05 else "infirmée"}
    # C8-V4 : différence d'écarts appariée par seed
    d4 = [(d["dlind-elu1"][18000][s] - d["smax"][18000][s]) - (d["dlind-elu1"][1800][s] - d["smax"][1800][s])
          for s in SEEDS]
    i4 = ic(d4)
    out["C8-V4"] = {"diff": i4[0], "ic": i4[1:],
                    "verdict": "confirmée (le signe de l'IC dit qui en profite)" if (i4[1] > 0 or i4[2] < 0) else "indéterminée (0 dans l'IC)"}

    for k, r in out.items():
        print(f"{k} : {r['diff']:+.4f} IC [{r['ic'][0]:+.4f} ; {r['ic'][1]:+.4f}] → {r['verdict']}")

    # comparaison avec l'analyse officielle si présente
    if os.path.exists("results/c8g_analyse.json"):
        ref = json.load(open("results/c8g_analyse.json"))
        ecarts = []
        for k, r in out.items():
            officiel = ref.get("verdicts", {}).get(k) or ref.get(k)
            if isinstance(officiel, dict) and "diff" in officiel:
                ecarts.append(abs(officiel["diff"] - r["diff"]))
                if "ic" in officiel and len(officiel["ic"]) == 2:
                    ecarts.append(abs(officiel["ic"][0] - r["ic"][0]))
                    ecarts.append(abs(officiel["ic"][1] - r["ic"][1]))
        if ecarts:
            e = max(ecarts)
            print(f"écart max vs analyse officielle : {e:.2e} → " + ("IDENTIQUE ✔" if e < 5e-4 else "ÉCART — à investiguer"))
            out["ecart_max_vs_analyse"] = e

    json.dump(out, open("results/c8g_verification.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("Écrit : results/c8g_verification.json")


if __name__ == "__main__":
    main()
