#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C11-bis — vérification indépendante (quantile de Student recalculé par intégration)."""
import json
import math

import numpy as np

CONFIGS = ["smax", "dlind-elu1", "dlind-const"]
SEEDS = [811, 822, 833]


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
    tb = {c: {s: json.load(open(f"results/c11b_{c}_s{s}/test.json"))["test_best"]
              for s in SEEDS} for c in CONFIGS}
    rec = {}
    for nom, (a, b) in {"C11B-V2": ("dlind-elu1", "smax"),
                        "C11B-V3": ("dlind-const", "dlind-elu1")}.items():
        rec[nom] = ic([tb[a][s] - tb[b][s] for s in SEEDS])
        print(f"{nom} recalculé : {rec[nom][0]:+.4f} IC [{rec[nom][1]:+.4f} ; {rec[nom][2]:+.4f}]")
    ref = json.load(open("results/c11b_analyse.json"))["verdicts"]
    ecart = max(max(abs(ref[nom]["diff"] - rec[nom][0]),
                    abs(ref[nom]["ic"][0] - rec[nom][1]),
                    abs(ref[nom]["ic"][1] - rec[nom][2])) for nom in ("C11B-V2", "C11B-V3"))
    print(f"écart max analyse↔vérification : {ecart:.2e} → "
          + ("IDENTIQUE ✔" if ecart < 5e-4 else "ÉCART — à investiguer"))


if __name__ == "__main__":
    main()
