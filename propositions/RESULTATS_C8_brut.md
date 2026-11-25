# Résultats bruts C8 (GitHub Actions)

> Sortie automatique de `scripts/analyse_c8_github.py`, à lire avec `PROTOCOLE_C8_GITHUB.md`.

```
C8 (GitHub) : 45/45 runs présents
commits des runs : ['46daced509'] ; code modifié : non

Perte moyenne (± écart type entre seeds) — poids choisis par la val (`best`), puis poids finaux
variante      budget  n         test best        test final  val best  œuvres best pas best (min–max)
smax            1800  5    1.8204 ± 0.0097    1.8204 ± 0.0097    1.8164       1.8818          1800-1800
smax            5400  5    1.4935 ± 0.0072    1.4935 ± 0.0072    1.4879       1.6277          5400-5400
smax           18000  5    1.3577 ± 0.0041    1.3580 ± 0.0042    1.3466       1.5287        17400-18000
dlind-elu1      1800  5    1.8352 ± 0.0020    1.8352 ± 0.0020    1.8340       1.8926          1800-1800
dlind-elu1      5400  5    1.5803 ± 0.0037    1.5803 ± 0.0037    1.5762       1.6922          5400-5400
dlind-elu1     18000  5    1.3942 ± 0.0026    1.3942 ± 0.0027    1.3865       1.5493        17400-18000
dlind-const     1800  5    1.8492 ± 0.0021    1.8492 ± 0.0021    1.8497       1.9081          1800-1800
dlind-const     5400  5    1.6236 ± 0.0039    1.6236 ± 0.0039    1.6234       1.7355          5400-5400
dlind-const    18000  5    1.4588 ± 0.0019    1.4591 ± 0.0025    1.4547       1.6101        17400-18000
n-gramme (val entière, fr1m) : 1.3402

Différences appariées par seed (< 0 : la première est meilleure)
  budget  1800 dlind-elu1 − smax          [test_best] +0.0148  IC [+0.0029 ; +0.0267]  0/5 seeds < 0
  budget  1800 dlind-const − smax         [test_best] +0.0287  IC [+0.0163 ; +0.0412]  0/5 seeds < 0
  budget  1800 dlind-const − dlind-elu1   [test_best] +0.0140  IC [+0.0112 ; +0.0167]  0/5 seeds < 0
  budget  5400 dlind-elu1 − smax          [test_best] +0.0868  IC [+0.0787 ; +0.0950]  0/5 seeds < 0
  budget  5400 dlind-const − smax         [test_best] +0.1301  IC [+0.1223 ; +0.1379]  0/5 seeds < 0
  budget  5400 dlind-const − dlind-elu1   [test_best] +0.0433  IC [+0.0377 ; +0.0489]  0/5 seeds < 0
  budget 18000 dlind-elu1 − smax          [test_best] +0.0364  IC [+0.0314 ; +0.0415]  0/5 seeds < 0
  budget 18000 dlind-const − smax         [test_best] +0.1011  IC [+0.0958 ; +0.1063]  0/5 seeds < 0
  budget 18000 dlind-const − dlind-elu1   [test_best] +0.0646  IC [+0.0622 ; +0.0671]  0/5 seeds < 0

--- Verdicts pré-enregistrés (métrique principale : test_best) ---
  C8-V0 (réplication de C7-V1 à 1 800 pas, dlind-elu1 − smax)                 → confirmée
  C8-V1 (dlind-elu1 bat smax à 18 000 pas)                                    → infirmée
  C8-V2 (dlind-elu1 bat smax à 5 400 pas)                                     → infirmée
  C8-V3 (dlind-const suffit à 18 000 pas, borne haute ≤ +0,05)                → infirmée
  C8-V4 (l'écart dlind-elu1 − smax évolue-t-il entre 1 800 et 18 000 pas ?)   → oui : l'avantage de smax grandit
  (évolution de l'écart : +0.0216, IC [+0.0098 ; +0.0335] ; + signifie que smax prend l'avantage quand le budget grandit)
  C8-V5 (description) meilleur réseau sur la val : smax|18000 = 1.3466 ; n-gramme 1.3402 ; écart +0.0064 nat (réseau derrière)

Écrit : /home/runner/work/imane_deep/imane_deep/results/c8g_analyse.json
```
