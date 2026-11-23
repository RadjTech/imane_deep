# Métriques CPU-first — définitions

> Les benchmarks GPU mesurent la qualité seule. Une recherche « sans GPU » doit mesurer **la qualité par unité de ressource CPU**. Ces définitions s'appliquent à toutes les entrées du journal.

## Qualité

| Symbole | Définition | Source |
|---|---|---|
| **Q** | perte val minimale (nats/caractère), protocole C1 (corpus 11 Ko, 1 800 pas, seed 1337) | `results/*/resume.json` |
| **Q@k** | perte val au pas k (pour comparer à budget d'entraînement fixé) | `historique.csv` |

## Coût mémoire d'inférence (le goulot CPU)

| Symbole | Définition | Valeurs actuelles |
|---|---|---|
| **M_inf** | octets d'état à conserver pour générer le token suivant | `smax` : KV-cache = 768 o/token/couche × T (croît) ; `lin` : 73 728 o/couche (constant) |
| **M@T** | M_inf à longueur T : `M@T = Σ_couches_smax 768×T + Σ_couches_lin 73 728` | smax×2 @T=4096 ≈ 6.3 Mo ; lin×2 ≈ 147 Ko (constant) |
| **M_inf/W** | M_inf rapporté au poids du modèle (240 480 params ≈ 942 Ko float32) | — |

## Vitesse

| Symbole | Définition |
|---|---|
| **V_train** | tokens/s d'entraînement (indicatif — dépend de l'implémentation NumPy, pas du concept) |
| **V_gen** | tokens/s de génération **batch 1** (le régime CPU réel) |

## Agrégations CPU-first (à publier par configuration)

1. **Qualité par bit** : Q vs bits/poids du checkpoint (float32 → int8 → ternaire) — cycle C4.
2. **Qualité par Go/s** : temps théorique de lecture des poids par token (`poids × octets / bande passante`) vs Q — la vraie monnaie de l'inférence CPU.
3. **Qualité par état** : Q vs M@T pour T ∈ {128, 1024, 4096} — c'est la métrique qui départage smax/lin/hybride.

## Mesures de référence (C2 : moyennes 3 seeds + profilage batch-1 réel)

**Qualité Q (val_min, moyenne ± std sur 3 seeds)** :

| Configuration | Q (val min) | Q@1800 (régime sur-appris) | M@T=4096 |
|---|---|---|---|
| `dlind,dlind` **ternaire** (C4) | 1.9154 ± 0.0125 | **1.9347 ± 0.026** | **0 (état 19 Ko) + poids 129 Ko packé (×7.3)** |
| `dlind,dlind` (C3) | **1.8976 ± 0.0131** | 2.0281 ± 0.047 | **0 (état 19 Ko constant)** |
| `smax,smax` | 2.1448 ± 0.018 | 3.1142 ± 0.0289 | ≈ 6,1 Mo (KV, croît) |
| `lin,smax` | 2.1652 ± 0.058 | 2.5515 ± 0.0804 | ≈ 3,1 Mo (croît ÷2) |
| `lin,lin` | 2.2480 ± 0.012 | 2.786 ± 0.0487 | **0 (état 19 Ko constant)** |

*Quantification a posteriori du checkpoint float (C4, bras B) : Δ +2.0853 nat (≈ hasard) — la ternarité doit être dans l'entraînement (BitNet confirmé sur notre stack).*

**V_gen batch-1 mesurée (tok/s, NumPy 2 CPU, ±20 % de bruit de timing)** :

| Contexte T | `dlind,dlind` ternaire (C4) | `dlind,dlind` (C3) | `smax,smax` | `lin,smax` | `lin,lin` |
|---|---|---|---|---|---|
| 256 | 2 571 | 2 404 | 2 404 | 2 394 | 2 494 |
| 1 024 | 2 532 | 1 789 | 1 341 | 1 824 | 2 652 |
| 4 096 | **2 628 (plat)** | **2 583 (plat)** | 588 | 948 | **2 693 (plat)** |

*Note ternaire : V_gen mesurée sur poids effectifs γ·Q en float32 (NumPy n'a pas de kernels entiers — le gain add-only de la littérature, 2.4–6×, est hors périmètre NumPy, consigné).*

KV-cache incrémental vs génération naïve : **×18,6** (140,7 → 2 610 tok/s). Le profil O(1) du linéaire et la décroissance O(T) du softmax sont maintenant des mesures, pas des prédictions.

*Note honnête : V_train NumPy n'est pas une métrique concept (les outer-products `lin` ne sont pas des matmuls BLAS). Les V_gen ci-dessus sont des mesures réelles du profileur (`results/profile_inf_*.json`).*
