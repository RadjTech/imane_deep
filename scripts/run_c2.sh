#!/usr/bin/env bash
# C2 — 8 runs restants (3 seeds × 3 configurations ; le point lin,smax/s1337 existe déjà en C1).
# Env identique à C1 (threads BLAS par défaut, séquentiel).
set -e
cd "$(dirname "$0")/.."
mkdir -p results/c2
git rev-parse HEAD > results/c2/GIT_COMMIT.txt
echo "protocole : $(cat results/c2/GIT_COMMIT.txt)"

run() {
  types="$1"; seed="$2"
  tag="c2_${types//,/-}_s${seed}"
  echo "=== RUN $tag ==="
  python3 -u experiments/hybrid_numpy.py --train --types "$types" --seed "$seed" \
      --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|1000|1800) "
}

# seeds nouveaux pour les 3 configurations
for seed in 2024 777; do
  run "smax,smax" "$seed"
  run "lin,lin"   "$seed"
  run "lin,smax"  "$seed"
done
# seed 1337 : les deux baselines avec le script unifié (test d'équivalence d'implémentation)
run "smax,smax" 1337
run "lin,lin"   1337

echo "C2 TERMINÉ" > results/c2/DONE.txt
echo "✔ tous les runs C2 sont terminés"
