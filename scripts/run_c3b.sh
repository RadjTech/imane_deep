#!/usr/bin/env bash
# C3b — re-mesurer les fondations (protocole : propositions/PROTOCOLE_C3b.md).
# 9 variantes × 5 seeds neufs, 2 processus à 1 thread, priorité basse, reprise possible.
#   bash scripts/run_c3b.sh            → garde-fous, runs, analyse
#   bash scripts/run_c3b.sh 11 22      → un sous-ensemble de seeds (les autres peuvent suivre plus tard)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1

SEEDS="${*:-11 22 33 44 55}"
VARIANTES="smax hybride-hist hybride-elu1 lin-hist lin-elu1 lin-hist-nk dlind-hist dlind-elu1 dlind-const"

$PY scripts/c3b_controles.py --check
$PY scripts/gardefou_conformite.py
$PY experiments/hybrid_numpy.py --check-grads
mkdir -p results/c3b
git rev-parse HEAD > results/c3b/GIT_COMMIT.txt
$PY scripts/lancer_parallele.py --cycle c3b --variantes --configs $VARIANTES --seeds $SEEDS --jobs 2
$PY scripts/analyse_c3b.py --incomplet
echo "✔ C3b : runs terminés (analyse ci-dessus ; verdicts seulement quand les 45 runs sont là)"
