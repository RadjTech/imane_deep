#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 — analyse PRÉ-ENREGISTRÉE (écrite avant la fin des runs, cf. protocole §3).
Lit les 9 points statistiques, produit moyennes ± std, victoires appariées,
test d'équivalence d'implémentation, et les verdicts H0/H1b selon les critères
annoncés dans HYPOTHESES.md. Aucune autre analyse n'est autorisée sans le
ré-écrire et le re-commiter (anti p-hacking).
Usage : python3 scripts/analyse_c2.py"""
import csv
import json
import os

R = "results"


def val_min_de(chemin_csv):
    vals = []
    with open(chemin_csv, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row[0] == "step":
                continue
            vals.append((int(row[0]), float(row[2])))
    vmin = min(vals, key=lambda t: t[1])
    return round(vmin[1], 4), vmin[0]


def point(config, seed):
    d = os.path.join(R, f"c2_{config.replace(',', '-')}_s{seed}")
    rj = os.path.join(d, "resume.json")
    if os.path.exists(rj):
        j = json.load(open(rj, encoding="utf-8"))
        return j["val_min"], j["val_min_step"]
    return val_min_de(os.path.join(d, "historique.csv"))


def main():
    seeds = [1337, 2024, 777]
    configs = ["smax,smax", "lin,lin", "lin,smax"]
    data = {}
    for c in configs:
        pts = {}
        for s in seeds:
            if c == "lin,smax" and s == 1337:
                # réutilisé de C1 (même script, même seed) — resume.json absent (bug corrigé depuis)
                pts[s] = val_min_de(os.path.join(R, "c1_hybride_lin_smax", "historique.csv"))
            else:
                pts[s] = point(c, s)
        data[c] = pts

    print("=" * 74)
    print("C2 — 3 seeds × 3 configurations (val_min, nats/caractère)")
    print("=" * 74)
    print(f"{'config':<12} | " + " | ".join(f"s{s}" for s in seeds) + " |  moyenne |   écart type")
    stats = {}
    for c in configs:
        vs = [data[c][s][0] for s in seeds]
        import statistics as st
        m, sd = st.mean(vs), st.stdev(vs)
        stats[c] = {"val_min_par_seed": {str(s): data[c][s] for s in seeds}, "mean": round(m, 4), "std": round(sd, 4)}
        ligne = " | ".join(f"{v:.4f}@{data[c][s][1]:>4}" for v, s in zip(vs, seeds))
        print(f"{c:<12} | {ligne} |  {m:.4f} |   {sd:.4f}")

    # test d'équivalence d'implémentation (script unifié vs scripts du cours, seed 1337)
    print("\n--- Équivalence d'implémentation (seed 1337, script unifié vs scripts du cours) ---")
    equiv = {}
    for c, base_csv in [("smax,smax", "baseline_smax_smax.csv"), ("lin,lin", "baseline_lin_lin.csv")]:
        ref = val_min_de(os.path.join(R, base_csv))[0]
        uni = data[c][1337][0]
        d_ = abs(uni - ref)
        equiv[c] = {"script_unifie": uni, "script_cours": ref, "delta": round(d_, 4)}
        print(f"{c:<12} : unifié {uni:.4f} vs cours {ref:.4f} → Δ = {d_:.4f} {'✔ (<5e-3)' if d_ < 5e-3 else '✘ (≥5e-3) — re-run requis'}")

    # verdicts pré-enregistrés
    print("\n--- Verdicts selon les critères annoncés (HYPOTHESES.md) ---")
    m_h, m_s = stats["lin,smax"]["mean"], stats["smax,smax"]["mean"]
    std_max = max(stats[c]["std"] for c in configs)
    h1b = m_h <= m_s
    print(f"H1b : mean(lin,smax)={m_h:.4f} ≤ mean(smax,smax)={m_s:.4f} ?  → {'✔ CONFIRMÉE (établie à 3 seeds)' if h1b else '✘ INFIRMÉE — C1 était un hasard de seed'}")
    if std_max < 0.015:
        seuil = round(2 * std_max, 4)
        print(f"H0  : std_max={std_max:.4f} < 0.015 → seuil d'interprétation abaissé à 2×std = {seuil}")
    elif std_max > 0.05:
        seuil = None
        print(f"H0  : std_max={std_max:.4f} > 0.05 → PLUS DE SEEDS requis avant toute conclusion fine")
    else:
        seuil = round(2 * std_max, 4)
        print(f"H0  : std_max={std_max:.4f} dans [0.015, 0.05] → seuil d'interprétation = 2×std = {seuil}")
    victoires = sum(1 for s in seeds if data["lin,smax"][s][0] < data["smax,smax"][s][0])
    ecart_moyen = round(m_s - m_h, 4)
    interpretable = (seuil is not None) and (abs(ecart_moyen) >= seuil)
    print(f"Victoires appariées lin,smax vs smax,smax : {victoires}/3 seeds ; écart moyen = {ecart_moyen:+.4f} nat "
          f"({'≥' if interpretable else '<'} seuil {seuil} → {'INTERPRÉTABLE' if interpretable else 'bruit'})")

    out = {"stats": stats, "equivalence": equiv,
           "verdicts": {"H1b": "confirmée" if h1b else "infirmée",
                        "H0_std_max": round(std_max, 4), "seuil_interpretation": seuil,
                        "victoires_appariees": victoires, "ecart_moyen": ecart_moyen,
                        "interpretable": interpretable}}
    os.makedirs(os.path.join(R, "c2"), exist_ok=True)
    with open(os.path.join(R, "c2", "analyse.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nÉcrit : results/c2/analyse.json")


if __name__ == "__main__":
    main()
