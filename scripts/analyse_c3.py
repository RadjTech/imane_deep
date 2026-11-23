#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 — analyse PRÉ-ENREGISTRÉE (écrite avant les runs).
Verdicts selon les critères annoncés dans HYPOTHESES.md (H2, H4b).
Usage : python3 scripts/analyse_c3.py"""
import json, os, statistics as st

R = "results"
def val_de(d):
    j = os.path.join(R, d, "resume.json")
    if os.path.exists(j):
        x = json.load(open(j, encoding="utf-8"))
        return x["val_min"], x["val_final"]
    raise FileNotFoundError(j)

def serie(prefix, seeds):
    return {s: val_de(f"{prefix}_s{s}") for s in seeds}

def main():
    seeds = [1337, 2024, 777]
    dl = serie("c3_dlind-dlind", seeds)
    lin = {s: val_de(f"c2_lin-lin_s{s}") for s in seeds}
    smx = {s: val_de(f"c2_smax-smax_s{s}") for s in seeds}
    print("=" * 76)
    print("C3 — delta rule (dlind,dlind) vs baselines C2 (moyennes ± std, 3 seeds)")
    print("=" * 76)
    print(f"{'config':<14} | " + " | ".join(f"s{s}" for s in seeds) + " | moyenne |  std")
    stats = {}
    for nom, d in [("dlind,dlind", dl), ("lin,lin", lin), ("smax,smax", smx)]:
        vs = [d[s][0] for s in seeds]
        m, sd = st.mean(vs), st.stdev(vs)
        stats[nom] = {"val_min": {str(s): d[s][0] for s in seeds}, "mean": round(m, 4), "std": round(sd, 4)}
        print(f"{nom:<14} | " + " | ".join(f"{v:.4f}" for v in vs) + f" | {m:.4f} | {sd:.4f}")
    print("--- val@1800 (régime sur-appris) ---")
    for nom, d in [("dlind,dlind", dl), ("lin,lin", lin), ("smax,smax", smx)]:
        vs = [d[s][1] for s in seeds]
        stats[nom]["val1800_mean"] = round(st.mean(vs), 4)
        print(f"{nom:<14} : " + " | ".join(f"{v:.4f}" for v in vs) + f"  (moy {st.mean(vs):.4f})")
    m_dl, m_lin, m_smx = stats["dlind,dlind"]["mean"], stats["lin,lin"]["mean"], stats["smax,smax"]["mean"]
    ecart = m_lin - m_smx
    rattrap = (m_lin - m_dl) / ecart if ecart > 0 else float("nan")
    h2 = m_dl <= 2.1964
    h4b = stats["dlind,dlind"]["val1800_mean"] <= stats["smax,smax"]["val1800_mean"] - 0.2
    print("--- Verdicts (critères annoncés) ---")
    print(f"H2  : mean(dlind)={m_dl:.4f} ≤ 2.1964 ? → {'✔ CONFIRMÉE' if h2 else '✘ INFIRMÉE'}"
          f" (rattrapage {100*rattrap:.0f}% de l'écart lin↔smax)")
    print(f"H4b : val@1800 mean(dlind)={stats['dlind,dlind']['val1800_mean']:.4f} ≤ "
          f"{stats['smax,smax']['val1800_mean'] - 0.2:.4f} ? → {'✔ CONFIRMÉE' if h4b else '✘ INFIRMÉE'}")
    out = {"stats": stats, "rattrapage_pct": round(100 * rattrap, 1),
           "verdicts": {"H2": "confirmée" if h2 else "infirmée", "H4b": "confirmée" if h4b else "infirmée"}}
    with open(os.path.join(R, "c3", "analyse.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("Écrit : results/c3/analyse.json")

if __name__ == "__main__":
    main()
