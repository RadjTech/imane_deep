#!/bin/bash
# C4 — H3 ternaire (BitNet b1.58) : bras A (entraînement DANS la contrainte {−1,0,+1}, STE)
# + bras B (quantification a posteriori des checkpoints float C3, sans ré-entraînement).
# À lancer APRÈS : --check-grads ✔ ET --check-ternary ✔ (protocole étape 4).
set -e
mkdir -p results/c4
git rev-parse HEAD > results/c4/GIT_COMMIT.txt

for seed in 1337 2024 777; do
  tag="c4_dlind-dlind_tern_s${seed}"
  echo "=== RUN $tag ==="
  python3 experiments/hybrid_numpy.py --train --ternary --types dlind,dlind \
    --seed "$seed" --max-steps 1800 --out-dir "results/$tag" 2>&1 | tee "results/$tag.log" | grep -E "HYBRIDE|RÉSUMÉ|^ *(0|1000|1800) "
done

for seed in 1337 2024 777; do
  echo "=== QUANTIFY-AFTER s${seed} (bras B) ==="
  python3 experiments/hybrid_numpy.py --quantize-after "results/c3_dlind-dlind_s${seed}/checkpoint.npz" \
    --types dlind,dlind --seed "$seed" --out-dir results/c4 2>&1 | tee "results/c4/quantize_after_s${seed}.log" | grep RÉSUMÉ
done

python3 experiments/hybrid_numpy.py --export-ternary results/c4_dlind-dlind_tern_s1337/checkpoint.npz \
  --types dlind,dlind --out-ternary results/c4/checkpoint_ternaire_s1337.npz 2>&1 | tee results/c4/export_ternaire.log

echo "C4 TERMINÉ" > results/c4/DONE.txt
echo "✔ runs C4 terminés"
