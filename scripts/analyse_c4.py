#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C4 — PROTOCOLE AMENDÉ, commité AVANT les runs (RESEARCH_LOG.md, « Amendement C4 »).

Remplace l'analyse de 2a976e8 (conservée dans l'historique git). Changements, tous avant tout run :
  - φ corrigée (ELU+1) dans les deux bras ; témoin float RE-MESURÉ dans le même lot (3 seeds)
    au lieu du 1.8976 de C3 (φ historique) ;
  - Δ_A et Δ_B mesurés au MÊME point et avec le MÊME eval (val fixe, toute la val, déterministe) ;
    la version 2a976e8 comparait val_min (≈ pas 900, éval tirée) à un checkpoint au pas 1 800
    (éval fixe, modèle sur-appris) ;
  - différences APPARIÉES par seed, écart type d'échantillon (ddof = 1), IC à 95 % de Student.

Bras (dlind,dlind, seeds {1337, 2024, 777}, 1 800 pas, tout le reste identique) :
  float    : results/c4float_dlind-dlind_s<seed>/
  ternaire : results/c4tern_dlind-dlind_s<seed>/   (entraîné DANS la contrainte {−1,0,+1})

Points de mesure, pour chaque seed :
  best  = checkpoint_best.npz (meilleure val fixe de CE bras pendant l'entraînement)
  final = checkpoint.npz (pas 1 800)
  Δ_A = val(ternaire entraîné) − val(float)
  Δ_B = val(float ternarisé a posteriori, sans ré-entraînement) − val(float)

H3 (critère unique, annoncé avant les runs) : mean(Δ_A) < mean(Δ_B) / 2 au point « best »
→ confirmée ; sinon → infirmée. Secondaires annoncées : Δ_A < 0.15 nat ; Δ_B > 0.30 nat.
Rapporté à titre de continuité : Δ_A sur val_min (éval tirée, comme C3).

Usage : python3 scripts/analyse_c4.py
"""
import json
import os
import statistics as st
import sys

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTATS = os.environ.get("C4_RESULTATS") or os.path.join(RACINE, "results")   # surchargeable pour un essai
sys.path.insert(0, os.path.join(RACINE, "experiments"))
import hybrid_numpy as hn  # noqa: E402

SEEDS = (1337, 2024, 777)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}            # Student bilatéral 95 %, par ddl = n-1
POINTS = (("best", "checkpoint_best.npz"), ("final", "checkpoint.npz"))
CLEFS = hn.CLEFS_MATMUL


def lire_run(bras, seed):
    d = os.path.join(RESULTATS, f"{bras}_dlind-dlind_s{seed}")
    r = json.load(open(os.path.join(d, "resume.json"), encoding="utf-8"))
    assert r["seed"] == seed and r["types"] == ["dlind", "dlind"], f"{d} : run inattendu"
    return d, r


def bilan(nom, x):
    """moyenne, écart type d'échantillon et IC 95 % (Student) d'une série appariée."""
    m, s = st.mean(x), (st.stdev(x) if len(x) > 1 else float("nan"))
    demi = T95.get(len(x) - 1, 1.96) * s / len(x) ** 0.5 if len(x) > 1 else float("inf")
    return {"nom": nom, "n": len(x), "par_seed": [round(v, 4) for v in x], "moyenne": round(m, 4),
            "ecart_type": round(s, 4), "ic95": [round(m - demi, 4), round(m + demi, 4)]}


def main():
    _, ids, _, itos = hn.charger_donnees(os.path.join(RACINE, "data", "corpus_fr.txt"))
    val = ids[int(0.9 * len(ids)):]
    runs = {b: {s: lire_run(b, s) for s in SEEDS} for b in ("c4float", "c4tern")}
    for s in SEEDS:
        assert not runs["c4float"][s][1]["ternary"] and runs["c4tern"][s][1]["ternary"], "bras inversés"
    phis = {r[1]["phi"] for b in runs.values() for r in b.values()}
    assert len(phis) == 1, f"φ différentes entre les bras : {phis}"
    hn.PHI = phis.pop()

    cfg = {"vocab_size": len(itos), "d_model": 96, "n_head": 4, "types": ["dlind", "dlind"], "block_size": 96}

    def evalue(ckpt, ternaire):
        P = {k: v for k, v in np.load(ckpt).items()}
        return hn.evaluer_fixe(hn.parametres_effectifs(P, cfg) if ternaire else P, cfg, val, 96)

    print("=" * 78)
    print(f"C4 (amendé) — ternaire sur dlind,dlind — φ = {hn.PHI} — val fixe, toute la val ({len(val) - 1} prédictions)")
    print("=" * 78)

    out = {"phi": hn.PHI, "seeds": list(SEEDS), "points": {}, "controle_coherence": [], "val_min_aleatoire": {}}
    res = {}
    for point, fichier in POINTS:
        f, a, b = [], [], []
        for s in SEEDS:
            df, rf = runs["c4float"][s]
            dt, rt = runs["c4tern"][s]
            vf = evalue(os.path.join(df, fichier), False)
            va = evalue(os.path.join(dt, fichier), True)
            vb = evalue(os.path.join(df, fichier), True)
            f.append(vf); a.append(va); b.append(vb)
            if point == "best":      # cohérence : éval en cours d'entraînement = éval post-hoc
                for nom, v, ref in (("float", vf, rf["val_fixe_min"]), ("ternaire", va, rt["val_fixe_min"])):
                    out["controle_coherence"].append({"seed": s, "bras": nom, "entrainement": ref, "post_hoc": v,
                                                      "ecart": abs(v - ref)})
        res[point] = (f, a, b)
        dA = [x - y for x, y in zip(a, f)]
        dB = [x - y for x, y in zip(b, f)]
        out["points"][point] = {"float": bilan("float", f), "ternaire": bilan("ternaire", a),
                                "quantifie_a_posteriori": bilan("quantifié", b),
                                "delta_A": bilan("Δ_A", dA), "delta_B": bilan("Δ_B", dB)}
        print(f"\n--- point « {point} » ---")
        print(f"{'seed':>6} | {'float':>8} | {'ternaire':>8} | {'Δ_A':>8} | {'a posteriori':>12} | {'Δ_B':>8}")
        for i, s in enumerate(SEEDS):
            print(f"{s:>6} | {f[i]:>8.4f} | {a[i]:>8.4f} | {dA[i]:>+8.4f} | {b[i]:>12.4f} | {dB[i]:>+8.4f}")
        bA, bB = out["points"][point]["delta_A"], out["points"][point]["delta_B"]
        print(f"Δ_A = {bA['moyenne']:+.4f} ± {bA['ecart_type']:.4f}  IC95 {bA['ic95']}")
        print(f"Δ_B = {bB['moyenne']:+.4f} ± {bB['ecart_type']:.4f}  IC95 {bB['ic95']}")

    pire = max(c["ecart"] for c in out["controle_coherence"])
    print(f"\ncohérence éval en cours d'entraînement = éval post-hoc (checkpoint_best) : écart max {pire:.2e}")
    assert pire < 1e-5, "l'évaluation post-hoc ne retrouve pas celle de l'entraînement : analyse invalide"

    # continuité avec C3 : val_min (éval tirée)
    vmin = {b: [runs[b][s][1]["val_min"] for s in SEEDS] for b in runs}
    dvmin = [x - y for x, y in zip(vmin["c4tern"], vmin["c4float"])]
    out["val_min_aleatoire"] = {"float": bilan("val_min float", vmin["c4float"]),
                                "ternaire": bilan("val_min ternaire", vmin["c4tern"]), "delta_A": bilan("Δ_A val_min", dvmin)}
    print(f"\nval_min (éval tirée, définition de C3) : float {st.mean(vmin['c4float']):.4f} | ternaire {st.mean(vmin['c4tern']):.4f} "
          f"| Δ_A {st.mean(dvmin):+.4f}")

    # verdicts
    best = out["points"]["best"]
    mA, mB = best["delta_A"]["moyenne"], best["delta_B"]["moyenne"]
    h3 = mA < mB / 2
    secA, secB = mA < 0.15, mB > 0.30
    print("\n--- Verdicts (critères annoncés avant les runs) ---")
    print(f"H3  : Δ_A ({mA:+.4f}) < Δ_B/2 ({mB / 2:+.4f}) au point « best » ? → {'CONFIRMÉE' if h3 else 'INFIRMÉE'}")
    print(f"sec : Δ_A < 0.15 ? {'oui' if secA else 'non'} | Δ_B > 0.30 ? {'oui' if secB else 'non'}")
    fin = out["points"]["final"]
    print(f"même critère au point « final » : Δ_A {fin['delta_A']['moyenne']:+.4f} vs Δ_B/2 "
          f"{fin['delta_B']['moyenne'] / 2:+.4f} → {'confirmée' if fin['delta_A']['moyenne'] < fin['delta_B']['moyenne'] / 2 else 'infirmée'}")
    sigma = max(best["float"]["ecart_type"], best["ternaire"]["ecart_type"])
    print(f"seuil H0 (2σ, σ = {sigma:.4f}) = {2 * sigma:.4f} nat ; |Δ_A| = {abs(mA):.4f} → "
          f"{'interprétable' if abs(mA) >= 2 * sigma else 'sous le seuil de bruit'}")

    # mémoire mesurée depuis le checkpoint ternaire (s1337, best)
    P = {k: v for k, v in np.load(os.path.join(runs["c4tern"][1337][0], "checkpoint_best.npz")).items()}
    total = sum(v.size for v in P.values())
    tern = sum(v.size for k, v in P.items() if any(k.endswith(c) for c in CLEFS))
    reste = total - tern
    n_ech = len({k.split(".")[0] for k in P if any(k.endswith(c) for c in CLEFS)}) * len(CLEFS)
    o_float = total * 4
    o_tern = int(tern * 2 / 8 + n_ech * 4 + reste * 4)
    out["memoire"] = {"parametres": total, "ternaires": tern, "reste_float32": reste, "octets_float": o_float,
                      "octets_ternaire_2bits": o_tern, "ratio": round(o_float / o_tern, 2)}
    print(f"mémoire : {o_float / 1024:.1f} Ko → {o_tern / 1024:.1f} Ko (×{o_float / o_tern:.2f}) ; "
          f"{reste} paramètres ({100 * reste / total:.1f} %) restent en float32")

    out["verdicts"] = {"H3_best": "confirmée" if h3 else "infirmée", "delta_A_best": mA, "delta_B_best": mB,
                       "secondaire_delta_A_lt_0.15": bool(secA), "secondaire_delta_B_gt_0.30": bool(secB)}
    chemin = os.path.join(RESULTATS, "c4", "analyse.json")
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as g:
        json.dump(out, g, indent=1, ensure_ascii=False)
    print("Écrit : results/c4/analyse.json")


if __name__ == "__main__":
    main()
