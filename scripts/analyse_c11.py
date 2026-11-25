#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C11 — PRÉ-ENREGISTRÉE (PROTOCOLE_C11.md, commité avant les runs).

test_best principal, différences appariées par seed, IC 95 % Student (n−1 ddl, ddof=1).
  C11-V1 : IC(dlind-elu1 − smax) @fr5m_v2/b5400 — tranché ou « indéterminé ».
  C11-V2 : IC(dlind-const − dlind-elu1), borne haute ≤ +0.05 — ou « indéterminé ».
  C11-V3 (descriptif) : meilleur réseau (val fr5m_v2) vs n-gramme fr5m_v2 (val).
  C11-V4 (descriptif) : chevauchement avec l'écart fr1m/b5400 de C8.
ATTENTION (§3 du protocole) : pertes fr1m ↔ fr5m_v2 incomparables en absolu.
"""
import json
import math
import os

import numpy as np

CONFIGS = ["smax", "dlind-elu1", "dlind-const"]
SEEDS = [711, 722, 733]
T95 = {2: 4.303}


def ic(diffs):
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    moy = float(x.mean())
    sd = float(x.std(ddof=1))
    demi = T95[n - 1] * sd / math.sqrt(n)
    return moy, moy - demi, moy + demi


def main():
    d = {c: {s: json.load(open(f"results/c11_{c}_s{s}/test.json")) for s in SEEDS} for c in CONFIGS}
    val = {c: {s: json.load(open(f"results/c11_{c}_s{s}/resume.json")) for s in SEEDS} for c in CONFIGS}
    print("=" * 78)
    print("C11 — fr5m_v2 (5.0 M car.), T=96, 5 400 pas, 3 seeds — principal : test_best")
    print("=" * 78)
    for c in CONFIGS:
        tb = np.array([d[c][s]["test_best"] for s in SEEDS])
        tf = np.array([d[c][s]["test_final"] for s in SEEDS])
        ve = np.array([val[c][s]["val_entiere_best"] for s in SEEDS])
        oe = np.array([d[c][s]["oeuvres_best"] for s in SEEDS])
        print(f"{c:12s} | test_best " + " ".join(f"{x:.4f}" for x in tb)
              + f" | moy {tb.mean():.4f} ± {tb.std(ddof=1):.4f}"
              + f" | test_final {tf.mean():.4f} | val {ve.mean():.4f} | œuvres_v2 {oe.mean():.4f}")

    e1 = [d["dlind-elu1"][s]["test_best"] - d["smax"][s]["test_best"] for s in SEEDS]
    i1 = ic(e1)
    v1v = "confirmée (la delta rule gagne)" if i1[2] < 0 else \
          ("infirmée (la classique gagne)" if i1[1] > 0 else "indéterminé (0 dans l'IC)")
    e2 = [d["dlind-const"][s]["test_best"] - d["dlind-elu1"][s]["test_best"] for s in SEEDS]
    i2 = ic(e2)
    v2v = "indéterminé (0 dans l'IC)" if (i2[1] <= 0 <= i2[2]) else \
          ("confirmée (borne haute " + f"{i2[2]:+.4f} ≤ +0.05)" if i2[2] <= 0.05 else "infirmée")

    ngram = json.load(open("results/c11_ngram_fr5m.json"))["perte_val_entiere"]
    meilleur_ve = min(CONFIGS, key=lambda c: np.mean([val[c][s]["val_entiere_best"] for s in SEEDS]))
    m_ve = float(np.mean([val[meilleur_ve][s]["val_entiere_best"] for s in SEEDS]))
    c8 = json.load(open("results/c8g_verification.json"))["C8-V2"]["ic"]
    chevauche = not (i1[2] < c8[0] or i1[1] > c8[1])

    print("--- Verdicts pré-enregistrés ---")
    print(f"C11-V1 dlind-elu1 − smax : {i1[0]:+.4f} IC [{i1[1]:+.4f} ; {i1[2]:+.4f}] → {v1v}")
    print(f"C11-V2 dlind-const − dlind-elu1 : {i2[0]:+.4f} IC [{i2[1]:+.4f} ; {i2[2]:+.4f}] → {v2v}")
    print(f"C11-V3 (descriptif) : val — meilleur {meilleur_ve} {m_ve:.4f} vs n-gramme fr5m_v2 {ngram:.4f}"
          f" → écart {m_ve - ngram:+.4f}")
    print(f"C11-V4 (descriptif) : IC fr5m_v2 vs écart fr1m/b5400 {c8[0]:+.4f}…{c8[1]:+.4f} → "
          + ("chevauchement : compatible" if chevauche else "disjoints : l'écart a changé avec les données"))

    out = {"cycle": "c11", "corpus": "data/fr5m_v2", "budget": 5400, "seeds": SEEDS,
           "test_best": {c: {str(s): d[c][s]["test_best"] for s in SEEDS} for c in CONFIGS},
           "test_final": {c: {str(s): d[c][s]["test_final"] for s in SEEDS} for c in CONFIGS},
           "val_entiere_best": {c: {str(s): val[c][s]["val_entiere_best"] for s in SEEDS} for c in CONFIGS},
           "oeuvres_best": {c: {str(s): d[c][s]["oeuvres_best"] for s in SEEDS} for c in CONFIGS},
           "ngram_val_entiere": ngram,
           "verdicts": {
               "C11-V1": {"diff": round(i1[0], 4), "ic": [round(i1[1], 4), round(i1[2], 4)], "verdict": v1v},
               "C11-V2": {"diff": round(i2[0], 4), "ic": [round(i2[1], 4), round(i2[2], 4)], "verdict": v2v},
               "C11-V3": {"meilleur": meilleur_ve, "val_moy": round(m_ve, 4), "ngram": ngram,
                          "ecart": round(m_ve - ngram, 4)},
               "C11-V4": {"chevauche_c8_fr1m": chevauche, "ic_c8_fr1m": [round(c8[0], 4), round(c8[1], 4)]}},
           "indetermine": "indéterminé" in (v1v, v2v)}
    json.dump(out, open("results/c11_analyse.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("Écrit : results/c11_analyse.json")


if __name__ == "__main__":
    main()
