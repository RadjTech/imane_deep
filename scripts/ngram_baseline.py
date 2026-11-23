#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Référence non neuronale : n-gramme de caractères (Kneser-Ney interpolé).

Même découpage 90/10 que hybrid_numpy.py et même protocole d'évaluation que les
modèles : fenêtres de T=96 caractères, contexte tronqué au début de la fenêtre,
moyenne sur toutes les positions. Les hyperparamètres (ordre, décompte d) sont
choisis sur les 10 % finaux du TRAIN : la validation n'est jamais consultée pour
régler quoi que ce soit. Aucune dépendance (ni numpy).

Usage : python3 scripts/ngram_baseline.py [--data data/corpus_fr.txt]
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict


def racine_depot():
    """Remonte jusqu'au dossier qui contient experiments/hybrid_numpy.py (marche depuis scripts/ ou propositions/scripts/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


def compter(txt, ordre):
    C = [defaultdict(Counter) for _ in range(ordre + 1)]
    for i, ch in enumerate(txt):
        for k in range(min(ordre, i) + 1):
            C[k][txt[i - k:i]][ch] += 1
    Cc = [defaultdict(Counter) for _ in range(ordre + 1)]     # comptes de continuation
    for k in range(ordre):
        for h, c in C[k + 1].items():
            for ch in c:
                Cc[k][h[1:]][ch] += 1
    return C, Cc


def proba(C, Cc, ordre, ctx, ch, d, V):
    p, kmax = 1.0 / V, min(ordre, len(ctx))
    for k in range(kmax + 1):
        h = ctx[len(ctx) - k:] if k else ""
        c = (C[k] if k == kmax else Cc[k]).get(h)
        if not c:
            break
        N, D = sum(c.values()), len(c)
        p = max(c.get(ch, 0) - d, 0) / N + d * D / N * p
    return p


def perte_fenetres(C, Cc, ordre, txt, d, V, T=96, pas=16):
    """nats/caractère, fenêtres déterministes (départ tous les `pas` caractères)."""
    departs = list(range(0, len(txt) - T - 1, pas)) + [len(txt) - T - 1]
    s = m = 0
    for a in departs:
        for j in range(1, T + 1):
            ctx = txt[a + max(0, j - ordre):a + j]
            s -= math.log(max(proba(C, Cc, ordre, ctx, txt[a + j], d, V), 1e-12))
            m += 1
    return s / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(racine_depot(), "data", "corpus_fr.txt"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t = open(a.data, encoding="utf-8").read()
    n = int(0.9 * len(t))
    tr, va, V = t[:n], t[n:], len(set(t))
    nd = int(0.9 * len(tr))
    tr0, dev = tr[:nd], tr[nd:]
    essais = []
    for o in (3, 4, 5, 6):
        C, Cc = compter(tr0, o)
        essais += [(perte_fenetres(C, Cc, o, dev, d, V), o, d) for d in (0.7, 0.8, 0.9)]
    _, o, d = min(essais)
    t0 = time.perf_counter()
    C, Cc = compter(tr, o)
    dt = time.perf_counter() - t0
    res = {"modele": f"Kneser-Ney {o + 1}-gramme caractere", "d": d, "choisi_sur": "10 % finaux du train",
           "perte_val_entiere": round(perte_fenetres(C, Cc, o, va, d, V), 4),
           "entrainement_s": round(dt, 2), "train": len(tr), "val": len(va), "vocab": V}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
