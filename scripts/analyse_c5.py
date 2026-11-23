#!/usr/bin/env python3
"""Analyse C5 — PRÉ-ENREGISTRÉE avant les runs (protocole étape 3, commité).

Comparaisons INTERNES au cycle (toutes les configs C5 : 1 200 pas, B=16, T=96,
d=96, H=4, mêmes données/optimiseur/seeds). Pas de comparaison directe aux
baselines C3/C4 (1 800 pas) — consigné.

Critères annoncés (HYPOTHESES.md) :
    H5  : mean(val_min)(dlind×4) ≤ mean(val_min)(smax×4) → confirmée ;
          écart < 2×std_max(C5) → « confirmée mais non distinguable du bruit » ; sinon infirmée.
    H6  : mean(val_min)(dlind×4 ternaire) − mean(val_min)(dlind×4 float) ≤ 0.05 → confirmée.
    H4c : mean(val@1200)(dlind×4 float) ≤ mean(val@1200)(smax×4) − 0.2 → confirmée.
"""
import glob
import json

import numpy as np

CONFIGS = {"dlind4": "dlind,dlind,dlind,dlind",
           "smax4": "smax,smax,smax,smax",
           "dlind4tern": "dlind×4 + ternaire"}


def charger(prefix):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/c5_{prefix}_s*/resume.json"))]
    assert len(runs) == 3, f"{prefix} : 3 seeds attendus, trouvé {len(runs)}"
    return runs


def main():
    data = {k: charger(k) for k in CONFIGS}
    stats = {}
    for k, runs in data.items():
        vmin = np.array([r["val_min"] for r in runs])
        vfin = np.array([r["val_final"] for r in runs])
        stats[k] = {"val_min": vmin, "val_1200": vfin,
                    "mean_vmin": float(vmin.mean()), "std_vmin": float(vmin.std()),
                    "mean_1200": float(vfin.mean()), "std_1200": float(vfin.std()),
                    "params": runs[0]["params"], "seeds": [r["seed"] for r in runs]}
    std_max = max(s["std_vmin"] for s in stats.values())
    seuil_bruit = 2 * std_max

    d = stats["dlind4"]
    s = stats["smax4"]
    t = stats["dlind4tern"]

    print("=" * 76)
    print("C5 — profondeur (4 couches, ≈464 K params, 1 200 pas) — comparaisons internes")
    print("=" * 76)
    for k in CONFIGS:
        st = stats[k]
        print(f"{k:11s} | val_min : " + " ".join(f"{v:.4f}" for v in st["val_min"])
              + f" | mean {st['mean_vmin']:.4f} ± {st['std_vmin']:.4f}"
              f" | val@1200 mean {st['mean_1200']:.4f} | params {st['params']:,}")
    print(f"seuil de bruit H0 interne : 2×std_max = {seuil_bruit:.4f} nat")

    # H5
    diff = d["mean_vmin"] - s["mean_vmin"]
    if d["mean_vmin"] <= s["mean_vmin"]:
        h5 = "confirmée" + (" (non distinguable du bruit)" if abs(diff) < seuil_bruit else "")
    else:
        h5 = "indistinguable (égalité au bruit)" if abs(diff) < seuil_bruit else "infirmée"
    # H6
    delta_tern = t["mean_vmin"] - d["mean_vmin"]
    h6 = "confirmée" if delta_tern <= 0.05 else "infirmée (coût notable à depth)"
    # H4c
    diff_reg = d["mean_1200"] - s["mean_1200"]
    h4c = "confirmée" if diff_reg <= -0.2 else "infirmée"

    print("--- Verdicts (critères annoncés) ---")
    print(f"H5  : dlind×4 ({d['mean_vmin']:.4f}) vs smax×4 ({s['mean_vmin']:.4f}) — "
          f"écart {diff:+.4f} → {h5}")
    print(f"H6  : Δ ternaire = {delta_tern:+.4f} nat (≤ 0.05 ?) → {h6}")
    print(f"H4c : val@1200 dlind×4 − smax×4 = {diff_reg:+.4f} (≤ −0.2 ?) → {h4c}")

    out = {
        "stats": {k: {"val_min": st["val_min"].tolist(), "val_1200": st["val_1200"].tolist(),
                      "mean_vmin": round(st["mean_vmin"], 4), "std_vmin": round(st["std_vmin"], 4),
                      "mean_1200": round(st["mean_1200"], 4), "params": st["params"],
                      "seeds": st["seeds"]} for k, st in stats.items()},
        "seuil_bruit": round(seuil_bruit, 4),
        "verdicts": {"H5": h5, "H6": h6, "H4c": h4c,
                     "delta_dlind_smax": round(float(diff), 4),
                     "delta_ternaire": round(float(delta_tern), 4),
                     "delta_regularisation": round(float(diff_reg), 4)},
    }
    with open("results/c5/analyse.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("Écrit : results/c5/analyse.json")


if __name__ == "__main__":
    main()
