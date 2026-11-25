#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse C8 (GitHub Actions) — PRÉ-ENREGISTRÉE avant les runs (propositions/PROTOCOLE_C8_GITHUB.md).

Runs : results/c8g_<variante>_b<budget>_s<seed>/ (3 variantes × 3 budgets × 5 seeds = 45).
Métrique principale : perte TEST avec les poids choisis par la val entière (`test_best`).
Comparaisons appariées par seed, IC 95 % de Student (n−1 ddl, ddof = 1) ; quantile recalculé (pas de table).

  python3 scripts/analyse_c8_github.py                 (verdicts ; refuse si des runs manquent)
  python3 scripts/analyse_c8_github.py --incomplet     (état sans verdict)
"""
import argparse
import json
import math
import os
import statistics as st

RACINE = os.environ.get("IMANE_RACINE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTES = ("smax", "dlind-elu1", "dlind-const")
BUDGETS = (1800, 5400, 18000)
SEEDS = (611, 622, 633, 644, 655)
EQUIV = 0.05


def quantile_t(p, df):
    """Quantile de Student : densité intégrée (trapèzes) et bissection, sans dépendance."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def dens(t):
        return c * (1 + t * t / df) ** (-(df + 1) / 2)

    def cdf(x, n=4000):
        pas = x / n
        return 0.5 + pas * (0.5 * dens(0) + sum(dens(i * pas) for i in range(1, n)) + 0.5 * dens(x))
    bas, haut = 0.0, 100.0
    for _ in range(80):
        m = (bas + haut) / 2
        bas, haut = (m, haut) if cdf(m) < p else (bas, m)
    return (bas + haut) / 2


def charger(racine):
    pts = {}
    for v in VARIANTES:
        for b in BUDGETS:
            for s in SEEDS:
                d = os.path.join(racine, f"c8g_{v}_b{b}_s{s}")
                fr, ft = os.path.join(d, "resume.json"), os.path.join(d, "test.json")
                if os.path.exists(fr) and os.path.exists(ft):
                    with open(fr, encoding="utf-8") as f:
                        r = json.load(f)
                    with open(ft, encoding="utf-8") as f:
                        t = json.load(f)
                    if r["seed"] != s or r["variante"] != v or r["budget"] != b:
                        raise SystemExit(f"{d} : seed, variante ou budget inattendu")
                    env = {}
                    fe = os.path.join(d, "ENV.json")
                    if os.path.exists(fe):
                        with open(fe, encoding="utf-8") as f:
                            env = json.load(f)
                    pts[v, b, s] = {**r, **{k: x for k, x in t.items() if k not in ("variante", "seed", "budget")}, "env": env}
    return pts


def apparier(pts, a, b_, budget, cle):
    """Différences a − b_ sur les seeds communs ; moyenne, écart type, IC 95 %, nombre de seeds où a < b_."""
    communs = [s for s in SEEDS if (a, budget, s) in pts and (b_, budget, s) in pts]
    if len(communs) < 2:
        return None
    d = [pts[a, budget, s][cle] - pts[b_, budget, s][cle] for s in communs]
    m, sd = st.mean(d), st.stdev(d)
    h = quantile_t(0.975, len(d) - 1) * sd / math.sqrt(len(d))
    return {"n": len(d), "moyenne": m, "ecart_type": sd, "ic95": [m - h, m + h], "negatifs": sum(x < 0 for x in d),
            "diff_par_seed": {str(s): round(x, 4) for s, x in zip(communs, d)}}


def verdict(r, attendu):
    """attendu : '<0' (a meilleure), '>0' (b meilleure), 'haut<=x' (non-infériorité)."""
    if r is None:
        return "indéterminée (données insuffisantes)"
    lo, hi = r["ic95"]
    if attendu == "<0":
        return "confirmée" if hi < 0 else ("infirmée" if lo > 0 else "indéterminée")
    if attendu == ">0":
        return "confirmée" if lo > 0 else ("infirmée" if hi < 0 else "indéterminée")
    if attendu.startswith("haut<="):
        s = float(attendu.split("<=")[1])
        return "confirmée" if hi <= s else ("infirmée" if lo > s else "indéterminée")
    raise ValueError(attendu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", default=os.path.join(RACINE, "results"))
    ap.add_argument("--md", default="", help="écrit aussi un compte rendu Markdown")
    ap.add_argument("--incomplet", action="store_true", help="affiche l'état sans verdict si des runs manquent")
    a = ap.parse_args()
    pts = charger(a.racine)
    attendus = len(VARIANTES) * len(BUDGETS) * len(SEEDS)
    complet = len(pts) == attendus
    sortie = []

    def p(x=""):
        sortie.append(x)
        print(x)
    p(f"C8 (GitHub) : {len(pts)}/{attendus} runs présents" + ("" if complet else " — INCOMPLET"))
    if not complet and not a.incomplet:
        manquants = [f"{v} b{b} s{s}" for v in VARIANTES for b in BUDGETS for s in SEEDS if (v, b, s) not in pts]
        p("Manquants : " + ", ".join(manquants[:15]) + (" …" if len(manquants) > 15 else ""))
        raise SystemExit("Verdicts refusés tant que les 45 runs ne sont pas tous là (--incomplet pour voir l'état).")
    commits = sorted({e["env"].get("commit", "?")[:10] for e in pts.values()})
    modif = [k for k, e in pts.items() if e["env"].get("code_modifie")]
    p(f"commits des runs : {commits} ; code modifié : {'non' if not modif else modif}")

    p("")
    p("Perte moyenne (± écart type entre seeds) — poids choisis par la val (`best`), puis poids finaux")
    p(f"{'variante':<12} {'budget':>7} {'n':>2} {'test best':>17} {'test final':>17} {'val best':>9} {'œuvres best':>12} {'pas best (min–max)':>18}")
    tableau = {}
    for v in VARIANTES:
        for b in BUDGETS:
            xs = [pts[v, b, s] for s in SEEDS if (v, b, s) in pts]
            if not xs:
                continue
            g = lambda c: (st.mean(x[c] for x in xs), st.stdev(x[c] for x in xs) if len(xs) > 1 else float("nan"))
            tb, tf, vb, ob = g("test_best"), g("test_final"), g("val_entiere_best"), g("oeuvres_best")
            pb = [x["pas_best"] for x in xs]
            tableau[f"{v}|{b}"] = {"n": len(xs), "test_best": tb, "test_final": tf, "val_best": vb, "oeuvres_best": ob,
                                   "pas_best": [min(pb), max(pb)]}
            plage = f"{min(pb)}-{max(pb)}"
            p(f"{v:<12} {b:>7} {len(xs):>2} {tb[0]:>9.4f} ± {tb[1]:.4f} {tf[0]:>9.4f} ± {tf[1]:.4f} {vb[0]:>9.4f} {ob[0]:>12.4f} {plage:>18}")

    ng = None
    fn = os.path.join(a.racine, "c7_ngram_fr1m.json")
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as f:
            ng = json.load(f)["perte_val_entiere"]
        p(f"n-gramme (val entière, fr1m) : {ng}")

    paires = []
    for b in BUDGETS:
        for nom, x, y in (("dlind-elu1 − smax", "dlind-elu1", "smax"), ("dlind-const − smax", "dlind-const", "smax"),
                          ("dlind-const − dlind-elu1", "dlind-const", "dlind-elu1")):
            for cle in ("test_best", "test_final"):
                paires.append((b, nom, x, y, cle))
    p("")
    p("Différences appariées par seed (< 0 : la première est meilleure)")
    res = {}
    for b, nom, x, y, cle in paires:
        r = apparier(pts, x, y, b, cle)
        res[f"{b}|{nom}|{cle}"] = r
        if r is not None and cle == "test_best":
            p(f"  budget {b:>5} {nom:<26} [{cle}] {r['moyenne']:+.4f}  IC [{r['ic95'][0]:+.4f} ; {r['ic95'][1]:+.4f}]  {r['negatifs']}/{r['n']} seeds < 0")

    def r(b, x, y):
        return res.get(f"{b}|{x} − {y}|test_best")
    # C8-V3 : évolution de l'écart avec le budget (apparié par seed)
    evol = None
    communs = [s for s in SEEDS if all((v, b, s) in pts for v in ("dlind-elu1", "smax") for b in (1800, 18000))]
    if len(communs) >= 2:
        ecarts = [(pts["dlind-elu1", 18000, s]["test_best"] - pts["smax", 18000, s]["test_best"])
                  - (pts["dlind-elu1", 1800, s]["test_best"] - pts["smax", 1800, s]["test_best"]) for s in communs]
        m, sd = st.mean(ecarts), st.stdev(ecarts)
        h = quantile_t(0.975, len(ecarts) - 1) * sd / math.sqrt(len(ecarts))
        evol = {"n": len(ecarts), "moyenne": m, "ic95": [m - h, m + h]}

    verdicts = {
        "C8-V0 (réplication de C7-V1 à 1 800 pas, dlind-elu1 − smax)": verdict(r(1800, "dlind-elu1", "smax"), ">0"),
        "C8-V1 (dlind-elu1 bat smax à 18 000 pas)": verdict(r(18000, "dlind-elu1", "smax"), "<0"),
        "C8-V2 (dlind-elu1 bat smax à 5 400 pas)": verdict(r(5400, "dlind-elu1", "smax"), "<0"),
        "C8-V3 (dlind-const suffit à 18 000 pas, borne haute ≤ +0,05)": verdict(r(18000, "dlind-const", "dlind-elu1"), f"haut<={EQUIV}"),
    }
    if evol is None:
        verdicts["C8-V4 (l'écart dlind-elu1 − smax évolue-t-il entre 1 800 et 18 000 pas ?)"] = "indéterminée (données insuffisantes)"
    else:
        lo, hi = evol["ic95"]
        verdicts["C8-V4 (l'écart dlind-elu1 − smax évolue-t-il entre 1 800 et 18 000 pas ?)"] = (
            "oui : l'avantage de smax grandit" if lo > 0 else "oui : dlind rattrape smax" if hi < 0 else "non établi (l'IC contient 0)")
    p("")
    p("--- Verdicts pré-enregistrés (métrique principale : test_best) ---" + ("" if complet else "  PROVISOIRES"))
    for k, v in verdicts.items():
        p(f"  {k:<75} → {v}")
    if evol is not None:
        p(f"  (évolution de l'écart : {evol['moyenne']:+.4f}, IC [{evol['ic95'][0]:+.4f} ; {evol['ic95'][1]:+.4f}] ; "
          f"+ signifie que smax prend l'avantage quand le budget grandit)")
    meilleurs = {f"{v}|{b}": st.mean(pts[v, b, s]["val_entiere_best"] for s in SEEDS if (v, b, s) in pts)
                 for v in VARIANTES for b in BUDGETS if any((v, b, s) in pts for s in SEEDS)}
    if ng is not None and meilleurs:
        k = min(meilleurs, key=meilleurs.get)
        p(f"  C8-V5 (description) meilleur réseau sur la val : {k} = {meilleurs[k]:.4f} ; n-gramme {ng} ; "
          f"écart {meilleurs[k] - ng:+.4f} nat (réseau {'devant' if meilleurs[k] < ng else 'derrière'})")
    if not complet:
        p("\n(Provisoire : il manque des runs.)")
        return
    chemin = os.path.join(a.racine, "c8g_analyse.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"tableau": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()} for k, d in tableau.items()},
                   "differences": res, "evolution_ecart_1800_18000": evol, "verdicts": verdicts, "ngram_val_entiere": ng,
                   "commits": commits}, f, indent=1, ensure_ascii=False)
    p(f"\nÉcrit : {chemin}")
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Résultats bruts C8 (GitHub Actions)\n\n> Sortie automatique de `scripts/analyse_c8_github.py`, à lire avec `PROTOCOLE_C8_GITHUB.md`.\n\n```\n"
                    + "\n".join(sortie) + "\n```\n")


if __name__ == "__main__":
    main()
