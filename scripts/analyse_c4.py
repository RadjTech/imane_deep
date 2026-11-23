#!/usr/bin/env python3
"""Analyse C4 — PRÉ-ENREGISTRÉE avant les runs (protocole étape 3, commité).

H3 (critère annoncé à l'ouverture du programme) :
    Δ_A (entraînement dans la contrainte ternaire) < Δ_B (quantification a posteriori) / 2
    → confirmée ; sinon → infirmée.
Attentes secondaires (énoncé H3) : Δ_A < 0.15 nat ; Δ_B > 0.30 nat.

Définitions (fixées avant les runs) :
    Δ_A = mean(val_min)(dlind,dlind ternaire, 3 seeds) − 1.8976
          (baseline C3 float, MÊME eval d'entraînement : 10 batchs val tirés)
    Δ_B = mean par seed de (val_ternaire − val_float) sur l'eval fixe déterministe
          (fenêtres non chevauchantes sur TOUTE la val), ckpts C3 ternarisés post-hoc.
"""
import glob
import json

import numpy as np

BASELINE_VAL_MIN = 1.8976          # C3 : dlind,dlind float, 3 seeds
BASELINE_STD = 0.0131
SEUIL_INTERP = 2 * 0.0131          # H0 : écarts < 2×std non interprétables
SUFFIXES_MATMUL = ("qkv.W", "proj.W", "fc1.W", "fc2.W")


def main():
    # ---- bras A : entraînement dans la contrainte ----
    runs = []
    for f in sorted(glob.glob("results/c4_dlind-dlind_tern_s*/resume.json")):
        r = json.load(open(f))
        runs.append(r)
    assert len(runs) == 3, f"3 seeds attendus, trouvé {len(runs)}"
    vals_min = np.array([r["val_min"] for r in runs])
    vals_1800 = np.array([r["val_final"] for r in runs])
    seeds = [r["seed"] for r in runs]
    mean_A, std_A = float(vals_min.mean()), float(vals_min.std())
    delta_A = mean_A - BASELINE_VAL_MIN

    # ---- bras B : quantification a posteriori ----
    bras_b = [json.load(open(f)) for f in sorted(glob.glob("results/c4/quantize_after_s*.json"))]
    assert len(bras_b) == 3, f"3 ckpts attendus, trouvé {len(bras_b)}"
    deltas_b = np.array([b["delta_ternaire"] for b in bras_b])
    delta_B = float(deltas_b.mean())

    # ---- verdicts (critères pré-enregistrés) ----
    h3 = delta_A < delta_B / 2
    sec_A = delta_A < 0.15
    sec_B = delta_B > 0.30

    # ---- mémoire mesurée depuis le checkpoint ternaire (s1337) ----
    P = dict(np.load("results/c4_dlind-dlind_tern_s1337/checkpoint.npz"))
    total = sum(v.size for v in P.values())
    tern = sum(v.size for k, v in P.items()
               if any(k.endswith(s) for s in SUFFIXES_MATMUL))
    reste = total - tern
    n_echelles = 2 * len(SUFFIXES_MATMUL)             # 2 blocs × 4 tenseurs
    octets_float = total * 4
    octets_ternaire = int(tern * 2 / 8 + n_echelles * 4 + reste * 4)

    print("=" * 76)
    print("C4 — ternaire (BitNet b1.58) sur base dlind,dlind — baselines C3 float")
    print("=" * 76)
    print("Bras A — entraînement DANS la contrainte (val_min, eval d'entraînement) :")
    for r in runs:
        print(f"  s{r['seed']} : {r['val_min']:.4f} (val@1800 {r['val_final']:.4f})")
    print(f"  mean {mean_A:.4f} ± {std_A:.4f}  →  Δ_A = {delta_A:+.4f} nat vs float C3 (1.8976)")
    print("Bras B — quantification a posteriori (eval fixe, apparié par ckpt) :")
    for b in bras_b:
        print(f"  {b['ckpt'].split('_s')[-1].split('/')[0]} : float {b['val_float_fixe']:.4f}"
              f" → ternaire {b['val_ternaire_fixe']:.4f}  (Δ {b['delta_ternaire']:+.4f})")
    print(f"  mean Δ_B = {delta_B:+.4f} nat")
    print("--- Verdicts (critères annoncés) ---")
    print(f"H3  : Δ_A ({delta_A:.4f}) < Δ_B/2 ({delta_B / 2:.4f}) ? "
          f"→ {'✔ CONFIRMÉE' if h3 else '✘ INFIRMÉE'}")
    print(f"sec : Δ_A < 0.15 ? {'✔' if sec_A else '✘'} ({delta_A:+.4f}) | "
          f"Δ_B > 0.30 ? {'✔' if sec_B else '✘'} ({delta_B:+.4f})")
    print(f"Mémoire (mesurée) : float {octets_float / 1024:.1f} Ko → ternaire packé "
          f"{octets_ternaire / 1024:.1f} Ko (×{octets_float / octets_ternaire:.1f}) ; "
          f"seuil d'interprétabilité H0 : {SEUIL_INTERP:.4f} nat")

    out = {
        "bras_A": {"seeds": seeds, "val_min": vals_min.tolist(),
                   "val_1800": vals_1800.tolist(), "mean": round(mean_A, 4),
                   "std": round(std_A, 4), "delta_A": round(delta_A, 4)},
        "bras_B": [{"ckpt": b["ckpt"], "val_float_fixe": b["val_float_fixe"],
                    "val_ternaire_fixe": b["val_ternaire_fixe"],
                    "delta": b["delta_ternaire"]} for b in bras_b],
        "delta_B": round(delta_B, 4),
        "memoire": {"params_total": total, "params_ternaire": tern,
                    "octets_float": octets_float, "octets_ternaire": octets_ternaire,
                    "ratio": round(octets_float / octets_ternaire, 2)},
        "verdicts": {"H3": "confirmée" if h3 else "infirmée",
                     "secondaire_delta_A_lt_0.15": bool(sec_A),
                     "secondaire_delta_B_gt_0.30": bool(sec_B)},
    }
    with open("results/c4/analyse.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("Écrit : results/c4/analyse.json")


if __name__ == "__main__":
    main()
