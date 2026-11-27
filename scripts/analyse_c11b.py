#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C11-bis — PRÉ-ENREGISTRÉE (PROTOCOLE_C11B.md, commité avant les runs).

test_best principal, différences appariées par seed (811/822/833), IC 95 % Student.
  C11B-V1 (descriptif) : écart meilleur réseau ↔ compteur (1.222) à b18000 vs 0.2517 à b5400.
  C11B-V2 : IC(dlind-elu1 − smax) — tranché ou « indéterminé ».
  C11B-V3 : IC(dlind-const − dlind-elu1), borne haute ≤ +0.05.
  C11B-V4 (descriptif) : test_best b18000 vs b5400 (moyennes, seeds non appariés).
"""
import json
import math
import os

import numpy as np

CONFIGS = ["smax", "dlind-elu1", "dlind-const"]
SEEDS = [811, 822, 833]
T95 = {2: 4.303}


def ic(diffs):
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    moy = float(x.mean())
    sd = float(x.std(ddof=1))
    demi = T95[n - 1] * sd / math.sqrt(n)
    return moy, moy - demi, moy + demi


def main():
    d = {c: {s: json.load(open(f"results/c11b_{c}_s{s}/test.json")) for s in SEEDS} for c in CONFIGS}
    val = {c: {s: json.load(open(f"results/c11b_{c}_s{s}/resume.json")) for s in SEEDS} for c in CONFIGS}
    lignes = []
    lignes.append("C11-bis (fr5m_v2, 18 000 pas, seeds 811/822/833) — principal : test_best")
    for c in CONFIGS:
        tb = [d[c][s]["test_best"] for s in SEEDS]
        oe = [d[c][s]["oeuvres_best"] for s in SEEDS]
        ve = [val[c][s]["val_entiere_best"] for s in SEEDS]
        pb = [val[c][s]["pas_best"] for s in SEEDS]
        lignes.append(f"{c:12s} test_best " + " ".join(f"{x:.4f}" for x in tb)
                      + f" | moy {np.mean(tb):.4f} ± {np.std(tb, ddof=1):.4f}"
                      + f" | val {np.mean(ve):.4f} | œuvres {np.mean(oe):.4f} | pas_best {pb}")

    e1 = [d["dlind-elu1"][s]["test_best"] - d["smax"][s]["test_best"] for s in SEEDS]
    i1 = ic(e1)
    v2v = ("confirmée (la delta rule gagne)" if i1[2] < 0
           else "infirmée (la classique gagne)" if i1[1] > 0
           else "indéterminé (0 dans l'IC)")
    e2 = [d["dlind-const"][s]["test_best"] - d["dlind-elu1"][s]["test_best"] for s in SEEDS]
    i2 = ic(e2)
    v3v = ("indéterminé (0 dans l'IC)" if i2[1] <= 0 <= i2[2]
           else "confirmée (borne haute " + f"{i2[2]:+.4f} ≤ +0.05)" if i2[2] <= 0.05
           else "infirmée")

    ngram = json.load(open("results/c11_ngram_fr5m.json"))["perte_val_entiere"]
    meilleur_ve = min(CONFIGS, key=lambda c: np.mean([val[c][s]["val_entiere_best"] for s in SEEDS]))
    m_ve = float(np.mean([val[meilleur_ve][s]["val_entiere_best"] for s in SEEDS]))
    ecart = m_ve - ngram
    b5400 = json.load(open("results/c11_analyse.json"))["verdicts"]["C11-V3"]["ecart"]

    lignes.append("")
    lignes.append("--- Verdicts pré-enregistrés ---")
    lignes.append(f"C11B-V1 (descriptif) : val — meilleur {meilleur_ve} {m_ve:.4f} vs compteur {ngram:.4f}"
                  f" → écart {ecart:+.4f} (à b5400 : +{b5400:.4f} → le budget "
                  + ("referme le trou" if ecart < b5400 else "ne referme pas le trou") + ")")
    lignes.append(f"C11B-V2 dlind-elu1 − smax : {i1[0]:+.4f} IC [{i1[1]:+.4f} ; {i1[2]:+.4f}] → {v2v}")
    lignes.append(f"C11B-V3 dlind-const − dlind-elu1 : {i2[0]:+.4f} IC [{i2[1]:+.4f} ; {i2[2]:+.4f}] → {v3v}")

    # C11B-V4 : test_best b18000 vs b5400 (seeds différents — descriptif)
    c11_test = {c: float(np.mean([json.load(open(f"results/c11_{c}_s{s}/test.json"))["test_best"]
                                  for s in (711, 722, 733)])) for c in CONFIGS}
    lignes.append("C11B-V4 (descriptif) : test_best b18000 vs b5400 (seeds non appariés)")
    for c in CONFIGS:
        m18 = float(np.mean([d[c][s]["test_best"] for s in SEEDS]))
        lignes.append(f"  {c:12s} : {m18:.4f} vs {c11_test[c]:.4f} (Δ {m18 - c11_test[c]:+.4f})")

    print("\n".join(lignes))
    out = {"cycle": "c11b", "corpus": "data/fr5m_v2", "budget": 18000, "seeds": SEEDS,
           "test_best": {c: {str(s): d[c][s]["test_best"] for s in SEEDS} for c in CONFIGS},
           "test_final": {c: {str(s): d[c][s]["test_final"] for s in SEEDS} for c in CONFIGS},
           "val_entiere_best": {c: {str(s): val[c][s]["val_entiere_best"] for s in SEEDS} for c in CONFIGS},
           "oeuvres_best": {c: {str(s): d[c][s]["oeuvres_best"] for s in SEEDS} for c in CONFIGS},
           "ngram_val_entiere": ngram,
           "verdicts": {
               "C11B-V1": {"meilleur": meilleur_ve, "val_moy": round(m_ve, 4), "ngram": ngram,
                           "ecart": round(ecart, 4), "ecart_b5400": b5400,
                           "budget_referme": bool(ecart < b5400)},
               "C11B-V2": {"diff": round(i1[0], 4), "ic": [round(i1[1], 4), round(i1[2], 4)], "verdict": v2v},
               "C11B-V3": {"diff": round(i2[0], 4), "ic": [round(i2[1], 4), round(i2[2], 4)], "verdict": v3v}},
           "indetermine": "indéterminé" in (v2v, v3v)}
    json.dump(out, open("results/c11b_analyse.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open("propositions/RESULTATS_C11B_brut.md", "w", encoding="utf-8") as f:
        f.write("# Résultats bruts C11-bis (GitHub Actions)\n\n> Sortie automatique de "
                "`scripts/analyse_c11b.py`, à lire avec `propositions/PROTOCOLE_C11B.md`.\n\n```\n")
        f.write("\n".join(lignes))
        f.write("\n```\n")
    print("Écrit : results/c11b_analyse.json + propositions/RESULTATS_C11B_brut.md")


if __name__ == "__main__":
    main()
