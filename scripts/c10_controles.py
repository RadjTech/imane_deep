#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C10 — contexte plus long : T = 256, lot de 6 (1 536 caractères/pas comme C8).

Réutilise la boucle exacte de scripts/c8_github.py SANS la modifier (import + surcharge
des constantes T et LOT du module) — même init, mêmes batchs (ré-échantillonnés à la
longueur 256), même optimiseur, cosinus sur --max-steps, éval val entière (fenêtres de
256, pas 16), checkpoint_best, TEST une seule fois à la fin (best ET final),
test_oeuvres_v2. Voir propositions/PROTOCOLE_C10.md (pré-enregistré).

  python3 scripts/c10_controles.py --variante smax --seed 711 --out-dir results/c10_smax_T256_s711
"""
import argparse
import json
import os
import sys

RACINE = os.environ.get("IMANE_RACINE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "experiments"))
sys.path.insert(0, os.path.join(RACINE, "scripts"))
import c8_github as c8g  # noqa: E402

# surcharge du module (PROTOCOLE_C10.md §2) : contexte 256, lot 6 → 1 536 car./pas
c8g.T = 256
c8g.LOT = 6


def main():
    ap = argparse.ArgumentParser(description="Run C10 (T=256, lot=6)")
    ap.add_argument("--variante", choices=list(c8g.VARIANTES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=5400)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--pas-eval", type=int, default=16)
    ap.add_argument("--data", default=os.path.join(RACINE, "data", "fr1m", "corpus.txt"))
    ap.add_argument("--test", default=os.path.join(RACINE, "data", "fr1m", "test.txt"))
    ap.add_argument("--oeuvres", default=os.path.join(RACINE, "data", "test_oeuvres_v2.txt"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, "resume.json")):
        raise SystemExit(f"{a.out_dir} est déjà terminé (on n'écrase pas)")
    os.makedirs(a.out_dir, exist_ok=True)
    env = c8g.infos_env()
    env.update({"cycle": "c10", "T": c8g.T, "lot": c8g.LOT,
                "protocole": "propositions/PROTOCOLE_C10.md"})
    with open(os.path.join(a.out_dir, "ENV.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=1, ensure_ascii=False)
    c8g.entrainer(a)


if __name__ == "__main__":
    main()
