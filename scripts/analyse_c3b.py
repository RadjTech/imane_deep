#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C3b — PRÉ-ENREGISTRÉE (écrite et commitée avant les runs ; propositions/PROTOCOLE_C3b.md).

Métrique principale : minimum de la perte sur la val ENTIÈRE, déterministe (`val_entiere_min`).
Toutes les comparaisons sont APPARIÉES par seed (différence variante A − variante B, même seed) ;
IC à 95 % de Student sur les différences. Seeds attendus : 11 22 33 44 55 (9 variantes × 5 seeds).

  H5  dlind-const − dlind-hist  : IC ⊂ [−0,05 ; +0,05]             → l'oubli exponentiel suffit
  H6  dlind-elu1 − dlind-hist   : IC entièrement > 0               → la φ historique aidait dlind
  H7  dlind-elu1 − smax         : IC entièrement < 0               → dlind reste devant smax avec φ correcte
  H8  lin-hist-nk − lin-hist    : IC ⊂ [−0,05 ; +0,05]             → normaliser les clés ne suffit pas
  H9  lin-elu1 : l'hybride rattrape ≥ la moitié de l'écart lin → smax (φ correcte)

Verdicts possibles : confirmée / infirmée / indéterminée (l'IC ne tranche pas → 8 seeds, cf. protocole).

Usage : python3 scripts/analyse_c3b.py   [--racine results] [--incomplet]
"""
import argparse
import json
import os
import statistics as st

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
VARIANTES = ("smax", "hybride-hist", "hybride-elu1", "lin-hist", "lin-elu1", "lin-hist-nk",
             "dlind-hist", "dlind-elu1", "dlind-const")
SEEDS = (11, 22, 33, 44, 55)
N_GRAMME = 1.6827            # Kneser-Ney, même val, même découpage (scripts/ngram_baseline.py)
EQUIV = 0.05


def racine_depot():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


def charger(racine):
    pts = {}
    for v in VARIANTES:
        for s in SEEDS:
            f = os.path.join(racine, f"c3b_{v}_s{s}", "resume.json")
            if os.path.exists(f):
                j = json.load(open(f, encoding="utf-8"))
                assert j["seed"] == s, f"{f} : seed inattendu"
                pts.setdefault(v, {})[s] = j
    return pts


def moyenne_sigma(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def apparier(pts, a, b, cle="val_entiere_min"):
    """Différences a − b sur les seeds communs, moyenne, écart type, IC 95 %."""
    communs = sorted(set(pts.get(a, {})) & set(pts.get(b, {})))
    if not communs:
        return None
    d = [pts[a][s][cle] - pts[b][s][cle] for s in communs]
    m, sd = moyenne_sigma(d)
    demi = T95.get(len(d) - 1, 1.96) * sd / len(d) ** 0.5 if len(d) > 1 else float("inf")
    return {"a": a, "b": b, "n": len(d), "diff_par_seed": {str(s): round(x, 4) for s, x in zip(communs, d)},
            "moyenne": m, "ecart_type": sd, "ic95": (m - demi, m + demi),
            "victoires": sum(x < 0 for x in d)}


def verdict_equivalence(r):
    if r is None or r["n"] < 2:
        return "indéterminée (données insuffisantes)"
    lo, hi = r["ic95"]
    if -EQUIV <= lo and hi <= EQUIV:
        return "confirmée"
    if lo > EQUIV or hi < -EQUIV:
        return "infirmée"
    return "indéterminée"


def verdict_signe(r, attendu):
    """attendu = '<0' ou '>0' : l'IC entier doit être du bon côté."""
    if r is None or r["n"] < 2:
        return "indéterminée (données insuffisantes)"
    lo, hi = r["ic95"]
    if attendu == ">0":
        return "confirmée" if lo > 0 else ("infirmée" if hi <= 0 else "indéterminée")
    return "confirmée" if hi < 0 else ("infirmée" if lo >= 0 else "indéterminée")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", default=os.path.join(racine_depot(), "results"))
    ap.add_argument("--incomplet", action="store_true", help="affiche aussi sans les 45 runs (aucun verdict)")
    a = ap.parse_args()
    pts = charger(a.racine)
    n = sum(len(d) for d in pts.values())
    complet = all(len(pts.get(v, {})) == len(SEEDS) for v in VARIANTES)
    print(f"C3b : {n}/{len(VARIANTES) * len(SEEDS)} runs présents{'' if complet else ' — INCOMPLET'}")
    if not complet and not a.incomplet:
        manquants = [f"{v} s{s}" for v in VARIANTES for s in SEEDS if s not in pts.get(v, {})]
        print("Manquants :", ", ".join(manquants[:12]), "…" if len(manquants) > 12 else "")
        raise SystemExit("Verdicts refusés tant que les 45 runs ne sont pas terminés (--incomplet pour voir l'état).")

    print(f"\nval entière, minimum pendant l'entraînement — référence n-gramme : {N_GRAMME} nat")
    print(f"{'variante':<13} {'n':>2} {'moyenne':>8} {'σ':>7} {'val @1800':>10} {'φ':>10}")
    tableau = {}
    for v in VARIANTES:
        if v not in pts:
            print(f"{v:<13} {'0':>2}")
            continue
        xs = [j["val_entiere_min"] for j in pts[v].values()]
        fin = [j["val_entiere_final"] for j in pts[v].values()]
        m, sd = moyenne_sigma(xs)
        phi = next(iter(pts[v].values())).get("phi", "?")
        tableau[v] = {"n": len(xs), "moyenne": m, "sigma": sd, "val_final_moyenne": st.mean(fin)}
        print(f"{v:<13} {len(xs):>2} {m:>8.4f} {sd:>7.4f} {st.mean(fin):>10.4f} {phi:>10}")

    paires = {
        "H5": ("dlind-const", "dlind-hist"),
        "H6": ("dlind-elu1", "dlind-hist"),
        "H7": ("dlind-elu1", "smax"),
        "H8": ("lin-hist-nk", "lin-hist"),
        "H9a (lin-elu1 − hybride-elu1)": ("lin-elu1", "hybride-elu1"),
        "H9b (lin-elu1 − smax)": ("lin-elu1", "smax"),
    }
    resultats = {}
    print("\nDifférences appariées (A − B), IC 95 % :")
    for nom, (x, y) in paires.items():
        r = apparier(pts, x, y)
        resultats[nom] = r
        if r is None:
            print(f"  {nom:<32} données absentes")
            continue
        print(f"  {nom:<32} {x} − {y} : {r['moyenne']:+.4f} ± {r['ecart_type']:.4f}  IC [{r['ic95'][0]:+.4f} ; "
              f"{r['ic95'][1]:+.4f}]  victoires {r['victoires']}/{r['n']}")

    verdicts = {
        "H5 (oubli exponentiel suffit)": verdict_equivalence(resultats["H5"]),
        "H6 (φ historique aidait dlind)": verdict_signe(resultats["H6"], ">0"),
        "H7 (dlind devant smax, φ correcte)": verdict_signe(resultats["H7"], "<0"),
        "H8 (normalisation des clés seule)": verdict_equivalence(resultats["H8"]),
    }
    r9a, r9b = resultats["H9a (lin-elu1 − hybride-elu1)"], resultats["H9b (lin-elu1 − smax)"]
    if r9b is None or r9a is None or r9b["n"] < 2:
        verdicts["H9 (H1 avec φ correcte)"] = "indéterminée (données insuffisantes)"
    elif r9b["ic95"][0] <= 0:
        verdicts["H9 (H1 avec φ correcte)"] = "indéterminée (l'écart lin → smax n'est pas établi)"
    else:
        ratio = r9a["moyenne"] / r9b["moyenne"]
        verdicts["H9 (H1 avec φ correcte)"] = (f"confirmée (rattrapage {100 * ratio:.0f} %)" if ratio >= 0.5
                                               else f"infirmée (rattrapage {100 * ratio:.0f} %)")

    print("\n--- Verdicts (critères du protocole) ---" + ("" if complet else "  PROVISOIRES, run incomplet"))
    for k, v in verdicts.items():
        print(f"  {k:<38} → {v}")

    if complet:
        chemin = os.path.join(a.racine, "c3b", "analyse.json")
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        json.dump({"tableau": tableau,
                   "differences": {k: (None if r is None else {**r, "ic95": list(r["ic95"])})
                                   for k, r in resultats.items()},
                   "verdicts": verdicts}, open(chemin, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\nÉcrit : {chemin}")


if __name__ == "__main__":
    main()
