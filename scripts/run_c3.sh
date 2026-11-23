#!/usr/bin/env bash
# C3 — delta rule : 3 seeds × dlind,dlind (protocole §3, HYPOTHESES.md H2/H4b)
set -e
cd "$(dirname "$0")/.."
mkdir -p results/c3
git rev-parse HEAD > results/c3/GIT_COMMIT.txt
for seed in 1337 2024 777; do
  tag="c3_dlind-dlind_s${seed}"
  echo "=== RUN $tag ==="
  python3 -u experiments/hybrid_numpy.py --train --types "dlind,dlind" --seed "$seed" \
      --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|1000|1800) "
done
echo "C3 TERMINÉ" > results/c3/DONE.txt
echo "✔ runs C3 terminés"
