#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C10 — PRÉ-ENREGISTRÉE (PROTOCOLE_C10.md, commité avant les runs).

Métrique principale : test_best (poids choisis par la val entière), différences
appariées par seed, IC 95 % de Student (n−1 ddl, ddof = 1).
  C10-V1 : IC(dlind-elu1 − smax) @T256/b5400 — tranché ou « indéterminé ».
  C10-V2 : IC(dlind-const − dlind-elu1) — borne haute ≤ +0.05 ou « indéterminé ».
  C10-V3 (descriptif) : chevauchement de l'IC avec l'écart C8 @T96/b5400.
  C10-V4 (descriptif) : meilleur réseau (val) vs n-gramme.
ATTENTION : les pertes T=256 ne se comparent pas en absolu aux pertes T=96 (§3 du
protocole) — seuls les écarts entre variantes au même T sont comparés.
"""
import json
import math
import os

import numpy as np

CONFIGS = ["smax", "dlind-elu1", "dlind-const"]
SEEDS = [s for s in ([711, 722, 733, 744, 755, 766])
          if __import__('os').path.exists(f"results/c10_smax_T256_s{s}/resume.json")]
T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}   # n−1 ddl


def ic(diffs):
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    moy = float(x.mean())
    sd = float(x.std(ddof=1))
    demi = T95[n - 1] * sd / math.sqrt(n)
    return moy, moy - demi, moy + demi


def main():
    d = {c: {s: json.load(open(f"results/c10_{c}_T256_s{s}/test.json")) for s in SEEDS}
         for c in CONFIGS}
    val = {c: {s: json.load(open(f"results/c10_{c}_T256_s{s}/resume.json")) for s in SEEDS}
           for c in CONFIGS}
    print("=" * 78)
    print("C10 — T=256, lot 6 (1 536 car./pas), 5 400 pas, 3 seeds — principal : test_best")
    print("=" * 78)
    for c in CONFIGS:
        tb = np.array([d[c][s]["test_best"] for s in SEEDS])
        tf = np.array([d[c][s]["test_final"] for s in SEEDS])
        ve = np.array([val[c][s]["val_entiere_best"] for s in SEEDS])
        oe = np.array([d[c][s]["oeuvres_best"] for s in SEEDS])
        pb = [val[c][s]["pas_best"] for s in SEEDS]
        print(f"{c:12s} | test_best " + " ".join(f"{x:.4f}" for x in tb)
              + f" | moy {tb.mean():.4f} ± {tb.std(ddof=1):.4f}"
              + f" | test_final {tf.mean():.4f} | val {ve.mean():.4f}"
              + f" | œuvres_v2 {oe.mean():.4f} | pas_best {pb}")

    e1 = [d["dlind-elu1"][s]["test_best"] - d["smax"][s]["test_best"] for s in SEEDS]
    i1 = ic(e1)
    v1v = "confirmée (la delta rule gagne)" if i1[2] < 0 else \
          ("infirmée (la classique gagne)" if i1[1] > 0 else "indéterminé (0 dans l'IC)")
    e2 = [d["dlind-const"][s]["test_best"] - d["dlind-elu1"][s]["test_best"] for s in SEEDS]
    i2 = ic(e2)
    v2v = "indéterminé (0 dans l'IC)" if (i2[1] <= 0 <= i2[2]) else \
          ("confirmée (borne haute " + f"{i2[2]:+.4f} ≤ +0.05)" if i2[2] <= 0.05 else "infirmée")

    c8 = json.load(open("results/c8g_verification.json"))["C8-V2"]["ic"]
    chevauche = not (i1[2] < c8[0] or i1[1] > c8[1])
    ngram = json.load(open("results/c7_ngram_fr1m.json"))["perte_val_entiere"]
    meilleur_ve = min(CONFIGS, key=lambda c: np.mean([val[c][s]["val_entiere_best"] for s in SEEDS]))
    m_ve = float(np.mean([val[meilleur_ve][s]["val_entiere_best"] for s in SEEDS]))

    print("--- Verdicts pré-enregistrés ---")
    print(f"C10-V1 dlind-elu1 − smax : {i1[0]:+.4f} IC [{i1[1]:+.4f} ; {i1[2]:+.4f}] → {v1v}")
    print(f"C10-V2 dlind-const − dlind-elu1 : {i2[0]:+.4f} IC [{i2[1]:+.4f} ; {i2[2]:+.4f}] → {v2v}")
    print(f"C10-V3 (descriptif) : IC T256 vs écart C8 T96 {c8[0]:+.4f}…{c8[1]:+.4f} → "
          + ("IC se chevauchent : pas de changement détecté" if chevauche else "IC disjoints : l'écart a changé avec le contexte"))
    print(f"C10-V4 (descriptif) : val — meilleur {meilleur_ve} {m_ve:.4f} vs n-gramme {ngram:.4f}"
          f" → écart {m_ve - ngram:+.4f}")

    out = {"cycle": "c10", "T": 256, "lot": 6, "budget": 5400, "seeds": SEEDS,
           "test_best": {c: {str(s): d[c][s]["test_best"] for s in SEEDS} for c in CONFIGS},
           "test_final": {c: {str(s): d[c][s]["test_final"] for s in SEEDS} for c in CONFIGS},
           "val_entiere_best": {c: {str(s): val[c][s]["val_entiere_best"] for s in SEEDS} for c in CONFIGS},
           "oeuvres_best": {c: {str(s): d[c][s]["oeuvres_best"] for s in SEEDS} for c in CONFIGS},
           "verdicts": {
               "C10-V1": {"diff": round(i1[0], 4), "ic": [round(i1[1], 4), round(i1[2], 4)], "verdict": v1v},
               "C10-V2": {"diff": round(i2[0], 4), "ic": [round(i2[1], 4), round(i2[2], 4)], "verdict": v2v},
               "C10-V3": {"chevauche_c8_T96": chevauche, "ic_c8_T96": [round(c8[0], 4), round(c8[1], 4)]},
               "C10-V4": {"meilleur": meilleur_ve, "val_moy": round(m_ve, 4), "ngram": ngram,
                          "ecart": round(m_ve - ngram, 4)}},
           "indetermine": "indéterminé" in (v1v, v2v)}
    json.dump(out, open("results/c10_analyse.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("Écrit : results/c10_analyse.json")


if __name__ == "__main__":
    main()
