#!/bin/bash
# C5 — profondeur : la delta rule et la ternarité survivent-elles à 4 couches ?
# Protocole §3 commité avant l'exécution. Garde-fous rejoués au début (étape 4,
# incluant les FD 4 couches ajoutés pour ce cycle). Comparaisons INTERNES au cycle :
# 1 200 pas pour TOUTES les configs (budget identique), B=16, T=96, d=96, H=4.
set -e
mkdir -p results/c5
git rev-parse HEAD > results/c5/GIT_COMMIT.txt

echo "=== GARDE-FOUS (protocole étape 4) ==="
python3 experiments/hybrid_numpy.py --check-grads
python3 experiments/hybrid_numpy.py --check-ternary

for seed in 1337 2024 777; do
  tag="c5_dlind4_s${seed}"
  echo "=== RUN $tag ==="
  python3 experiments/hybrid_numpy.py --train --types dlind,dlind,dlind,dlind \
    --seed "$seed" --max-steps 1200 --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|600|1200) "
done

for seed in 1337 2024 777; do
  tag="c5_smax4_s${seed}"
  echo "=== RUN $tag ==="
  python3 experiments/hybrid_numpy.py --train --types smax,smax,smax,smax \
    --seed "$seed" --max-steps 1200 --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|600|1200) "
done

for seed in 1337 2024 777; do
  tag="c5_dlind4tern_s${seed}"
  echo "=== RUN $tag ==="
  python3 experiments/hybrid_numpy.py --train --ternary --types dlind,dlind,dlind,dlind \
    --seed "$seed" --max-steps 1200 --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|600|1200) "
done

echo "C5 TERMINÉ" > results/c5/DONE.txt
echo "✔ runs C5 terminés"
