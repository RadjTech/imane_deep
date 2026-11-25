#!/bin/bash
# Série C10 : 9 runs (3 variantes × 3 seeds), 2 processus à 1 thread, priorité basse.
# Reprise après coupure : saute les runs dont resume.json existe.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
for V in smax dlind-elu1 dlind-const; do
  for S in 711 722 733; do
    D="results/c10_${V}_T256_s${S}"
    if [ -f "$D/resume.json" ]; then echo "sauté $D (déjà terminé)"; continue; fi
    mkdir -p "$D"
    echo "lancé $D"
    nice -n 10 python3 -u scripts/c10_controles.py --variante "$V" --seed "$S" \
      --max-steps 5400 --out-dir "$D" > "$D/train.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  done
done
wait
echo "Série C10 terminée"
