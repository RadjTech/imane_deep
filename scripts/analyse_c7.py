#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C7 — PRÉ-ENREGISTRÉE avant les runs (PROTOCOLE_C7.md, commis avant).

Critères (perte TEST, poids finaux, une seule évaluation ; différences appariées
par seed ; IC 95 % de Student, n−1 ddl, écart type ddof = 1) :
  C7-V1 : IC(dlind-elu1 − smax) entièrement < 0 → confirmée.
  C7-V2 : borne haute de IC(dlind-const − dlind-elu1) ≤ +0.05 → confirmée.
  C7-V3 : signe de l'IC(dlind-elu1 − dlind-hist) : >0 φ historique aide ;
          <0 φ corrigée meilleure ; 0 dedans → « on ne sait pas ».
  C7-V4 : descriptif — meilleur réseau (val entière) vs n-gramme (même val).
Un IC qui ne tranche pas → « indéterminée » et passage à 8 seeds (amendement)
avant toute conclusion.
"""
import glob
import json
import math
import os

import numpy as np

T95 = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}   # n−1 ddl
CONFIGS = ["smax", "lin-hist", "dlind-hist", "dlind-elu1", "dlind-const"]
SEEDS = [101, 202, 303, 404, 505]
CYCLE = "c7a"


def charger(metric="test"):
    """metric ∈ test, test_oeuvres, val_entiere_min, val_final — dict config → seed → valeur."""
    d = {}
    for c in CONFIGS:
        d[c] = {}
        for s in SEEDS:
            chemin_r = f"results/{CYCLE}_{c}_s{s}/resume.json"
            if metric in ("test", "test_oeuvres"):
                chemin = f"results/{CYCLE}_{c}_s{s}/test.json"
                if not os.path.exists(chemin):
                    raise SystemExit(f"il manque {chemin} — évaluer le test d'abord")
                d[c][s] = json.load(open(chemin))[metric]
            else:
                d[c][s] = json.load(open(chemin_r))[metric]
    return d


def ic(diffs):
    """diffs : liste des différences appariées par seed → (moyenne, bas, haut)."""
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    moy = float(x.mean())
    sd = float(x.std(ddof=1))
    t = T95[n - 1]
    demi = t * sd / math.sqrt(n)
    return moy, moy - demi, moy + demi


def verdict_ic(moy, bas, haut, cible):
    if cible == "<0":
        return "confirmée" if haut < 0 else ("indéterminée" if bas < 0 <= haut else "infirmée")
    if cible.startswith("haut<="):
        s = float(cible.split("<=")[1])
        return "confirmée" if haut <= s else "infirmée"


def main():
    d = {}
    for m in ("test", "test_oeuvres", "val_entiere_min", "val_final"):
        d[m] = charger(m)

    print("=" * 78)
    print(f"C7 — corpus fr1m, 5 variantes × {len(SEEDS)} seeds — métrique principale : TEST")
    print("=" * 78)
    for c in CONFIGS:
        vals = np.array([d["test"][c][s] for s in SEEDS])
        ve = np.array([d["val_entiere_min"][c][s] for s in SEEDS])
        v1800 = np.array([d["val_final"][c][s] for s in SEEDS])
        oe = np.array([d["test_oeuvres"][c][s] for s in SEEDS])
        print(f"{c:12s} | test : " + " ".join(f"{v:.4f}" for v in vals)
              + f" | moy {vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
              + f" | val_ent min {ve.mean():.4f} | val@1800 {v1800.mean():.4f}"
              + f" | test_oeuvres {oe.mean():.4f}")

    print("--- Verdicts pré-enregistrés ---")
    elab = []
    for s in SEEDS:
        elab.append(d["test"]["dlind-elu1"][s] - d["test"]["smax"][s])
    v1 = ic(elab)
    v1v = verdict_ic(*v1, "<0")
    print(f"C7-V1 dlind-elu1 − smax : {v1[0]:+.4f} IC [{v1[1]:+.4f} ; {v1[2]:+.4f}] → {v1v}")

    elab2 = [d["test"]["dlind-const"][s] - d["test"]["dlind-elu1"][s] for s in SEEDS]
    v2 = ic(elab2)
    v2v = verdict_ic(*v2, "haut<=0.05")
    print(f"C7-V2 dlind-const − dlind-elu1 : {v2[0]:+.4f} IC [{v2[1]:+.4f} ; {v2[2]:+.4f}] → {v2v}")

    elab3 = [d["test"]["dlind-elu1"][s] - d["test"]["dlind-hist"][s] for s in SEEDS]
    v3 = ic(elab3)
    if v3[1] > 0:
        v3v = "φ historique aide encore"
    elif v3[2] < 0:
        v3v = "la φ corrigée est meilleure"
    else:
        v3v = "on ne sait pas (0 dans l'IC)"
    print(f"C7-V3 dlind-elu1 − dlind-hist : {v3[0]:+.4f} IC [{v3[1]:+.4f} ; {v3[2]:+.4f}] → {v3v}")

    ngram = json.load(open("results/c7_ngram_fr1m.json"))["perte_val_entiere"]
    meilleur_ve = min(CONFIGS, key=lambda c: np.mean([d["val_entiere_min"][c][s] for s in SEEDS]))
    m_ve = float(np.mean([d["val_entiere_min"][meilleur_ve][s] for s in SEEDS]))
    print(f"C7-V4 (descriptif) : val entière — meilleur réseau {meilleur_ve} {m_ve:.4f}"
          f" vs n-gramme {ngram:.4f} → le réseau {('fait mieux' if m_ve < ngram else 'ne fait pas mieux')} que le n-gramme")

    indetermine = any(v in ("indéterminée", "on ne sait pas (0 dans l'IC)") for v in (v1v, v2v, v3v))
    if indetermine:
        print("→ au moins un IC ne tranche pas : « indéterminée », passage à 8 seeds requis (amendement) avant conclusion.")

    out = {
        "cycle": CYCLE, "corpus": "data/fr1m", "seeds": SEEDS,
        "test": {c: {str(s): d["test"][c][s] for s in SEEDS} for c in CONFIGS},
        "test_oeuvres": {c: {str(s): d["test_oeuvres"][c][s] for s in SEEDS} for c in CONFIGS},
        "val_entiere_min": {c: {str(s): d["val_entiere_min"][c][s] for s in SEEDS} for c in CONFIGS},
        "val_final": {c: {str(s): d["val_final"][c][s] for s in SEEDS} for c in CONFIGS},
        "moyennes": {c: {"test": round(float(np.mean([d['test'][c][s] for s in SEEDS])), 4),
                          "test_std": round(float(np.std([d['test'][c][s] for s in SEEDS], ddof=1)), 4),
                          "val_entiere_min": round(float(np.mean([d['val_entiere_min'][c][s] for s in SEEDS])), 4),
                          "val_final": round(float(np.mean([d['val_final'][c][s] for s in SEEDS])), 4),
                          "test_oeuvres": round(float(np.mean([d['test_oeuvres'][c][s] for s in SEEDS])), 4)}
                     for c in CONFIGS},
        "ngram_val_entiere": ngram,
        "verdicts": {
            "C7-V1": {"diff": round(v1[0], 4), "ic": [round(v1[1], 4), round(v1[2], 4)], "verdict": v1v},
            "C7-V2": {"diff": round(v2[0], 4), "ic": [round(v2[1], 4), round(v2[2], 4)], "verdict": v2v},
            "C7-V3": {"diff": round(v3[0], 4), "ic": [round(v3[1], 4), round(v3[2], 4)], "verdict": v3v},
            "C7-V4": {"meilleur_reseau": meilleur_ve, "val_entiere_min_moy": round(m_ve, 4),
                      "ngram": ngram, "reseau_mieux_que_ngram": m_ve < ngram},
        },
        "indeterminee_8_seeds_requis": indetermine,
    }
    with open(f"results/{CYCLE}_analyse.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"Écrit : results/{CYCLE}_analyse.json")


if __name__ == "__main__":
    main()
