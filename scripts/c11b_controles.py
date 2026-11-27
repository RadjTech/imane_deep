#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C11-bis — fr5m_v2, 18 000 pas (PROTOCOLE_C11B.md, pré-enregistré).

Réutilise la boucle exacte de scripts/c8_github.py sans la modifier (comme
c11_controles.py, dont c'est le jumeau à budget long : T=96, lot=16, pas-eval 96).

  python3 scripts/c11b_controles.py --variante smax --seed 811 --out-dir results/c11b_smax_s811
"""
import argparse
import json
import os
import sys

RACINE = os.environ.get("IMANE_RACINE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "experiments"))
sys.path.insert(0, os.path.join(RACINE, "scripts"))
import c8_github as c8g  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run C11-bis (fr5m_v2, 18 000 pas)")
    ap.add_argument("--variante", choices=list(c8g.VARIANTES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=18000)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--pas-eval", type=int, default=96)
    ap.add_argument("--data", default=os.path.join(RACINE, "data", "fr5m_v2", "corpus.txt"))
    ap.add_argument("--test", default=os.path.join(RACINE, "data", "fr5m_v2", "test.txt"))
    ap.add_argument("--oeuvres", default=os.path.join(RACINE, "data", "test_oeuvres_v2.txt"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, "resume.json")):
        raise SystemExit(f"{a.out_dir} est déjà terminé (on n'écrase pas)")
    os.makedirs(a.out_dir, exist_ok=True)
    env = c8g.infos_env()
    env.update({"cycle": "c11b", "T": c8g.T, "lot": c8g.LOT, "corpus": "data/fr5m_v2",
                "protocole": "propositions/PROTOCOLE_C11B.md"})
    with open(os.path.join(a.out_dir, "ENV.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=1, ensure_ascii=False)
    c8g.entrainer(a)


if __name__ == "__main__":
    main()
