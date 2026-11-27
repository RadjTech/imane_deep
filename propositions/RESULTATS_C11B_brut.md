# Résultats bruts C11-bis (GitHub Actions)

> Sortie automatique de `scripts/analyse_c11b.py`, à lire avec `propositions/PROTOCOLE_C11B.md`.

```
C11-bis (fr5m_v2, 18 000 pas, seeds 811/822/833) — principal : test_best
smax         test_best 1.2981 1.3028 1.3016 | moy 1.3008 ± 0.0024 | val 1.2886 | œuvres 1.4587 | pas_best [18000, 18000, 18000]
dlind-elu1   test_best 1.3601 1.3613 1.3551 | moy 1.3588 ± 0.0033 | val 1.3468 | œuvres 1.5055 | pas_best [18000, 18000, 18000]
dlind-const  test_best 1.4406 1.4447 1.4382 | moy 1.4412 ± 0.0033 | val 1.4303 | œuvres 1.5829 | pas_best [18000, 18000, 18000]

--- Verdicts pré-enregistrés ---
C11B-V1 (descriptif) : val — meilleur smax 1.2886 vs compteur 1.2220 → écart +0.0666 (à b5400 : +0.2517 → le budget referme le trou)
C11B-V2 dlind-elu1 − smax : +0.0580 IC [+0.0474 ; +0.0686] → infirmée (la classique gagne)
C11B-V3 dlind-const − dlind-elu1 : +0.0824 IC [+0.0784 ; +0.0863] → infirmée
C11B-V4 (descriptif) : test_best b18000 vs b5400 (seeds non appariés)
  smax         : 1.3008 vs 1.4849 (Δ -0.1841)
  dlind-elu1   : 1.3588 vs 1.5758 (Δ -0.2170)
  dlind-const  : 1.4412 vs 1.6231 (Δ -0.1820)
```
