#!/bin/bash
# Amendement C10 n°1 : seeds 744/755/766 pour smax et dlind-elu1 (T=256), 2 processus.
cd "$(dirname "$0")/.." || exit 1
for S in 744 755 766; do
  for V in smax dlind-elu1; do
    D="results/c10_${V}_T256_s${S}"
    if [ -f "$D/resume.json" ]; then echo "sauté $D"; continue; fi
    mkdir -p "$D"
    echo "lancé $D"
    nice -n 10 python3 -u scripts/c10_controles.py --variante "$V" --seed "$S" \
      --max-steps 5400 --out-dir "$D" > "$D/train.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  done
done
wait
echo "Amendement C10 terminé"
