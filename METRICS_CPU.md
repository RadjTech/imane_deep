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

## Mesures de référence actuelles (seed 1337, 2 CPU, NumPy)

| Configuration | Q (val min) | M@T=96 | M@T=4096 |
|---|---|---|---|
| `smax,smax` | **2.1214** | ≈ 147 Ko | ≈ 6.3 Mo (croît) |
| `lin,smax` (C1) | **2.1265** | ≈ 110 Ko | ≈ 3.2 Mo (croît ×0.5) |
| `lin,lin` | 2.2349 | **≈ 147 Ko constant** | **≈ 147 Ko constant** |

*Note honnête : V_train NumPy n'est pas une métrique concept (les outer-products `lin` ne sont pas des matmuls BLAS). V_gen batch-1 est la mesure qui compte — elle sera produite par C2 avec le profiler d'inférence.*
