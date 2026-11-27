#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Liste des essais C11-bis pour GitHub Actions (matrice=<json> pour $GITHUB_OUTPUT).

  petit-essai : 2 essais de 200 pas (dossiers c11bessai_*), pour vérifier la chaîne.
  complet     : les 9 essais de PROTOCOLE_C11B.md (3 variantes × seeds 811, 822, 833,
                18 000 pas), dossiers results/c11b_<variante>_s<seed>.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c8_github import VARIANTES  # noqa: E402

SEEDS = (811, 822, 833)
BUDGET = 18000


def essais(mode):
    if mode == "petit-essai":
        return [{"variante": v, "seed": SEEDS[0], "budget": 200, "tag": f"c11bessai_{v}_s{SEEDS[0]}"} for v in ("smax", "dlind-const")]
    if mode == "complet":
        ordre = sorted(VARIANTES, key=lambda v: {"dlind-elu1": 0, "dlind-const": 1, "smax": 2}[v])
        return [{"variante": v, "seed": s, "budget": BUDGET, "tag": f"c11b_{v}_s{s}"} for v in ordre for s in SEEDS]
    raise SystemExit("mode attendu : petit-essai ou complet")


if __name__ == "__main__":
    print("matrice=" + json.dumps({"include": essais(sys.argv[1] if len(sys.argv) > 1 else "")}, separators=(",", ":")))
