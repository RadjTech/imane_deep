#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agrégateur générique de cycle : lit tous les results/<prefixe>_<config>_s<seed>/
et produit moyennes ± écart type, comparaisons APPARIÉES par seed et verdict.

Usage : python3 scripts/agreger.py --prefixes c2 c3 --ref smax,smax [--md sortie.md] [--json sortie.json]

Règle appliquée (RESEARCH_PROTOCOL.md) : n < 3 seeds = « indication » ; un écart est
dit « établi » seulement si l'intervalle à 95 % de la différence appariée exclut 0
(Student, n-1 degrés de liberté — à 3 seeds il est large, et c'est voulu).
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st


def racine_depot():
    """Remonte jusqu'au dossier qui contient experiments/hybrid_numpy.py (marche depuis scripts/ ou propositions/scripts/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def lire(dossier, metrique="val_min"):
    """(meilleure val, val finale, train final) depuis resume.json, sinon depuis historique.csv."""
    rj = os.path.join(dossier, "resume.json")
    if os.path.exists(rj):
        j = json.load(open(rj, encoding="utf-8"))
        if metrique not in j:
            raise SystemExit(f"{rj} : pas de champ {metrique} (runs antérieurs : utiliser --metrique val_min)")
        return j[metrique], j[metrique.replace("_min", "_final")], j["train_final"]
    lignes = [r for r in csv.reader(open(os.path.join(dossier, "historique.csv"), encoding="utf-8")) if r[0] != "step"]
    vals = [float(r[2]) for r in lignes]
    return min(vals), vals[-1], float(lignes[-1][1])


def collecter(racine, prefixes, metrique):
    pts = {}
    for p in prefixes:
        for d in sorted(glob.glob(os.path.join(racine, f"{p}_*_s*"))):
            m = re.match(rf"{re.escape(p)}_(.+)_s(\d+)$", os.path.basename(d))
            if m and os.path.isdir(d):
                pts.setdefault(m.group(1), {})[int(m.group(2))] = lire(d, metrique)
    return pts


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", default=os.path.join(racine_depot(), "results"), help="dossier des résultats")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--ref", default="smax,smax")
    ap.add_argument("--metrique", choices=["val_min", "val_entiere_min"], default="val_min",
                    help="val_entiere_min : val complète déterministe (c3b_controles.py)")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    a.ref = a.ref.replace(",", "-")          # "smax,smax" et "smax-smax" sont acceptés
    pts = collecter(a.racine, a.prefixes, a.metrique)
    if not pts:
        raise SystemExit(f"Aucun run trouvé sous {a.racine}/ pour {a.prefixes}")
    L = ["| Configuration | seeds | val_min (moy ± σ) | val finale (moy ± σ) | surapprentissage |", "|---|---|---|---|---|"]
    out = {"configs": {}, "comparaisons": {}}
    for c, d in sorted(pts.items(), key=lambda kv: ms([v[0] for v in kv[1].values()])[0]):
        vm, vf = ms([v[0] for v in d.values()]), ms([v[1] for v in d.values()])
        L.append(f"| `{c}` | {len(d)} | {vm[0]:.4f} ± {vm[1]:.4f} | {vf[0]:.4f} ± {vf[1]:.4f} | +{vf[0] - vm[0]:.2f} |")
        out["configs"][c] = {"n": len(d), "val_min": vm, "val_final": vf, "par_seed": {str(s): v[0] for s, v in sorted(d.items())}}
    if a.ref in pts:
        L += ["", f"Différences appariées par seed, contre `{a.ref}` (négatif = meilleur) :", "",
              "| Configuration | n appariés | Δ val_min | IC 95 % | victoires | statut |", "|---|---|---|---|---|---|"]
        for c, d in pts.items():
            communs = sorted(set(d) & set(pts[a.ref]))
            if c == a.ref or not communs:
                continue
            diff = [d[s][0] - pts[a.ref][s][0] for s in communs]
            m, sd = ms(diff)
            n = len(diff)
            demi = T95.get(n - 1, 1.96) * sd / n ** 0.5 if n > 1 else float("inf")
            statut = "indication (n < 3)" if n < 3 else ("établi" if abs(m) > demi else "non établi")
            L.append(f"| `{c}` | {n} | {m:+.4f} | [{m - demi:+.4f} ; {m + demi:+.4f}] | {sum(x < 0 for x in diff)}/{n} | {statut} |")
            out["comparaisons"][c] = {"ref": a.ref, "n": n, "delta": m, "ic95": [m - demi, m + demi], "statut": statut}
    texte = "\n".join(L)
    print(texte)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(texte + "\n")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
