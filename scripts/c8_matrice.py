#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Liste des essais C8 pour GitHub Actions (une ligne « matrice=<json> » à écrire dans $GITHUB_OUTPUT).

  petit-essai : 2 essais très courts (200 pas) pour vérifier la chaîne, rien n'est conclu.
  complet     : les 45 essais du protocole, les plus longs d'abord (3 budgets × 3 variantes × 5 seeds).
Les listes viennent de c8_github.py : elles ne se changent pas à la main.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c8_github import BUDGETS, SEEDS, VARIANTES  # noqa: E402


def essais(mode):
    if mode == "petit-essai":
        return [{"variante": v, "seed": SEEDS[0], "budget": 200} for v in ("smax", "dlind-const")]
    if mode == "complet":
        return [{"variante": v, "seed": s, "budget": b} for b in BUDGETS for v in VARIANTES for s in SEEDS]
    raise SystemExit("mode attendu : petit-essai ou complet")


if __name__ == "__main__":
    print("matrice=" + json.dumps({"include": essais(sys.argv[1] if len(sys.argv) > 1 else "")}, separators=(",", ":")))
