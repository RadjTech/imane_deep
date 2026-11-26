#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C10 — vérification indépendante (quantile de Student recalculé par intégration)."""
import json
import math
import os

import numpy as np

CONFIGS = ["smax", "dlind-elu1", "dlind-const"]
SEEDS_P = [711, 722, 733, 744, 755, 766]
SEEDS = {c: [s for s in SEEDS_P
             if __import__('os').path.exists(f"results/c10_{c}_T256_s{s}/resume.json")]
         for c in CONFIGS}


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


def main():
    tb = {c: {s: json.load(open(f"results/c10_{c}_T256_s{s}/test.json"))["test_best"]
              for s in SEEDS[c]} for c in CONFIGS}
    rec = {}
    for nom, (a, b) in {"C10-V1": ("dlind-elu1", "smax"),
                        "C10-V2": ("dlind-const", "dlind-elu1")}.items():
        communs = sorted(set(SEEDS[a]) & set(SEEDS[b]))
        rec[nom] = ic([tb[a][s] - tb[b][s] for s in communs])
        print(f"  ({nom} : {len(communs)} seeds communs {communs})")
        print(f"{nom} recalculé : {rec[nom][0]:+.4f} IC [{rec[nom][1]:+.4f} ; {rec[nom][2]:+.4f}]")

    ref = json.load(open("results/c10_analyse.json"))["verdicts"]
    ecart = max(max(abs(ref[nom]["diff"] - rec[nom][0]),
                    abs(ref[nom]["ic"][0] - rec[nom][1]),
                    abs(ref[nom]["ic"][1] - rec[nom][2])) for nom in ("C10-V1", "C10-V2"))
    print(f"écart max analyse↔vérification : {ecart:.2e} → "
          + ("IDENTIQUE ✔" if ecart < 5e-4 else "ÉCART — à investiguer"))


if __name__ == "__main__":
    main()
