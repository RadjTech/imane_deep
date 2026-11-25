#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Liste des essais C11 pour GitHub Actions (une ligne « matrice=<json> » pour $GITHUB_OUTPUT).

  petit-essai : 2 essais de 200 pas (dossiers c11essai_*), pour vérifier la chaîne ; rien n'est conclu.
  complet     : les 9 essais de propositions/PROTOCOLE_C11.md (3 variantes × seeds 711, 722, 733, 5 400 pas),
                dossiers results/c11_<variante>_s<seed> comme scripts/c11_serie.sh.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c8_github import VARIANTES  # noqa: E402

SEEDS = (711, 722, 733)
BUDGET = 5400


def essais(mode):
    if mode == "petit-essai":
        return [{"variante": v, "seed": SEEDS[0], "budget": 200, "tag": f"c11essai_{v}_s{SEEDS[0]}"} for v in ("smax", "dlind-const")]
    if mode == "complet":
        ordre = sorted(VARIANTES, key=lambda v: {"dlind-elu1": 0, "dlind-const": 1, "smax": 2}[v])   # les plus longs d'abord
        return [{"variante": v, "seed": s, "budget": BUDGET, "tag": f"c11_{v}_s{s}"} for v in ordre for s in SEEDS]
    raise SystemExit("mode attendu : petit-essai ou complet")


if __name__ == "__main__":
    print("matrice=" + json.dumps({"include": essais(sys.argv[1] if len(sys.argv) > 1 else "")}, separators=(",", ":")))
