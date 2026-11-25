# Résultats C9 — le coût : vitesse et mémoire en génération (2026-11-25)

> Étape 3 de `PROMPT_SUITE.md`. Poids : checkpoints C7 s101 (les mesures de vitesse et de
> mémoire ne dépendent pas de la valeur des poids ; la qualité définitive viendra de C8).
> Note sur la consigne : la mission indiquait `--phi elu1`, mais chaque checkpoint est
> profilé avec **la φ de son entraînement** (option prudente — le garde-fou du profileur
> refuse sinon) : `smax` → φ historique, `dlind-elu1` → φ elu1, `dlind-const` → φ elu1
> (sans effet : la variante const n'utilise ni clés ni requêtes).
> Garde-fous : smax 2.15e-05 ✔, dlind-elu1 4.77e-06 ✔, dlind-const 3.81e-06 (double garde :
> parallèle local vs `hn.forward` installé, incrémental vs parallèle) ✔.

## 1. Tableau unique — qualité / vitesse / mémoire (batch 1, NumPy 2.3.5, 2 cœurs)

| Variante | Qualité : test fr1m (C7, provisoire) | V_gen @T=96 | V_gen @T=256 | V_gen @T=1024 | V_gen @T=4096 | Mémoire d'inférence @T=4096 |
|---|---|---|---|---|---|---|
| `smax` (φ hist) | **1.8223** | 4 793 | 3 888 | 2 133 | 899 | KV **6 144 Ko** (croît) |
| `dlind-elu1` | 1.8392 | 3 610 | 3 802 | 3 810 | **3 791 (plat)** | état 19 Ko, KV 0 |
| `dlind-const` | 1.8495 | 2 628* | 5 293 | 5 443 | **5 513 (plat)** | **état 0.8 Ko**, KV 0 |
| *n-gramme (réf.)* | *1.3402 (val)* | — | — | — | — | *297 Mo RAM (mesuré C7)* |

*V_gen @T=96 de dlind-const : mesure plus bruitée (150 tokens seulement à ce point) — voir
`results/profile_inf_dlindc-dlindc.json`. V_gen @T=96 des autres : `results/profile_inf_c9_*.json`.

**Coût en pour cent de perte par rapport à `smax`** (test C7, provisoire) :
`dlind-elu1` **+0.93 %** · `dlind-const` **+1.49 %** · (`lin-hist` +18.6 %, rappel).

## 2. Lecture

1. **À contexte court (≤ 96, le régime d'entraînement), `smax` est la plus rapide** (~4 800-5 100 tok/s) : son attention par scores est simple en NumPy. Mais son coût explose avec le contexte : **÷5.3 entre T=96 et T=4096** (4 793 → 899 tok/s) et 6,1 Mo de KV-cache.
2. **À contexte long, la delta rule gagne** : `dlind-elu1` est plat (~3 800 tok/s, ×4.2 vs smax @4096) et `dlind-const` est plat **et plus rapide que smax partout au-delà de T≈250** (~5 500 tok/s @4096, ×6.1 vs smax).
3. **La mémoire d'état : 6 144 Ko (smax) → 19 Ko (dlind) → 0.8 Ko (dlind-const)** — facteur 7 680 entre les deux extrêmes. L'état de `dlind-const` tient dans un L1 de cache processeur.
4. **Le compromis qualité/coût se lit ainsi** (chiffres C7 provisoires, à re-mesurer avec C8) : `dlind-const` coûte +1.49 % de perte et rapporte ×6.1 de vitesse à long contexte et ÷7 680 de mémoire d'état. Si C8 confirme que l'écart de qualité reste ~+0.02-0.03 nat, le compromis est nettement favorable pour tout usage à contexte long ou mémoire contrainte.
5. **Honnêteté NumPy** : ces vitesses sont celles des formes float32 de NumPy — le gain « multiplications → additions entières » de la littérature BitNet (2.4-6×) n'est pas mesuré ici (kernels entiers hors périmètre) ; l'avantage de `dlind-const` vient de la simplicité de sa mise à jour (pas de clés, pas de requêtes, pas de normalisation) et de son état minuscule, pas de kernels spéciaux.
6. **Périmètre** : batch 1 ; le prefill suit le même classement (4 612 / 4 120 / 5 489 tok/s @T=4096) ; machine 2 cœurs partagée avec rien d'autre au moment des mesures.

## 3. Ce qui manque

- La colonne qualité est **provisoire (C7, poids finaux à 1 800 pas)** — C8 (18 000 pas, checkpoint_best) la remplacera dès `results/c8g_analyse.json` disponible.
- `profiler_const.py` mesure une implémentation NumPy de la forme incrémentale vectorielle ; un profilage réel du coût par composante (qkv matmul dominant) n'est pas fait.
