# Résultats du cycle C3b — fondations re-mesurées (24/11/2026)

> Protocole `PROTOCOLE_C3b.md` (commit `7c61e39`, 00:56 ; mis à jour en `109319f`, 01:04) et analyse `scripts/analyse_c3b.py` (`109319f`)
> commités **avant** le premier run (01:05).
> 45 runs = 9 variantes × 5 seeds neufs (11, 22, 33, 44, 55), 1 800 pas, code non modifié depuis le commit.
>
> **Attention aux noms.** Ici H5 à H9 sont celles de `PROTOCOLE_C3b.md`. Dans `HYPOTHESES.md`, H5 et H6 désignent
> **autre chose** (C5 : profondeur, ternaire). Pour toute recopie, écrire `C3b-H5` … `C3b-H9`.

## 1. Résultats (perte sur la val entière, minimum pendant l'entraînement)

| Variante | s11 | s22 | s33 | s44 | s55 | moyenne | écart type | val @1800 | pas du min |
|---|---|---|---|---|---|---|---|---|---|
| `smax` | 2.1887 | 2.1654 | 2.1741 | 2.1374 | 2.1411 | **2.1613** | 0.0219 | 3.1032 | 600 |
| `hybride-hist` | 2.1807 | 2.1588 | 2.1069 | 2.1445 | 2.1758 | **2.1533** | 0.0297 | 2.5365 | 700–800 |
| `hybride-elu1` | 2.2137 | 2.1861 | 2.2036 | 2.1742 | 2.2245 | **2.2004** | 0.0204 | 3.0425 | 500–800 |
| `lin-hist` | 2.2644 | 2.2489 | 2.2526 | 2.2468 | 2.2588 | **2.2543** | 0.0073 | 2.7569 | 400–700 |
| `lin-elu1` | 2.2408 | 2.2347 | 2.2382 | 2.2449 | 2.2146 | **2.2347** | 0.0118 | 3.1427 | 500–700 |
| `lin-hist-nk` | 2.2533 | 2.2641 | 2.2598 | 2.2560 | 2.2778 | **2.2622** | 0.0096 | 2.5180 | 500–800 |
| `dlind-hist` (= C3) | 1.9055 | 1.9250 | 1.8962 | 1.9120 | 1.8718 | **1.9021** | 0.0199 | 2.0238 | 900–1000 |
| `dlind-elu1` | 1.9929 | 1.9899 | 1.9980 | 1.9901 | 2.0126 | **1.9967** | 0.0094 | 2.4471 | 600–700 |
| `dlind-const` | 1.8429 | 1.8801 | 1.8758 | 1.8671 | 1.8297 | **1.8591** | 0.0219 | 1.9168 | 1000–1300 |

Référence non neuronale, même val : n-gramme Kneser-Ney **1,6827** (`scripts/ngram_baseline.py`, 0,3 s).
**Aucune des 9 variantes ne fait mieux que le n-gramme.**

## 2. Verdicts (critères pré-enregistrés, différences appariées par seed, IC 95 % de Student)

| # | Comparaison | Différence moyenne | IC 95 % | Verdict |
|---|---|---|---|---|
| C3b-H5 | `dlind-const` − `dlind-hist` | −0,0430 (const meilleur 5/5) | [−0,0617 ; −0,0243] | **indéterminée** (la bande ±0,05 n'est pas respectée) |
| C3b-H6 | `dlind-elu1` − `dlind-hist` | +0,0946 | [+0,0585 ; +0,1307] | **confirmée** : la φ historique aidait `dlind` |
| C3b-H7 | `dlind-elu1` − `smax` | −0,1646 (5/5) | [−0,1976 ; −0,1316] | **confirmée** : `dlind` reste devant `smax` avec φ correcte |
| C3b-H8 | `lin-hist-nk` − `lin-hist` | +0,0079 | [−0,0066 ; +0,0224] | **confirmée** : normaliser les clés ne suffit pas |
| C3b-H9 | rattrapage de l'écart `lin` → `smax` (φ correcte) | 47 % (seuil 50 %) | voir § 3 | **infirmée** par la règle, à la limite |

## 3. Ce que l'on peut dire, et ne pas dire

**Établi (5 seeds neufs, critères fixés à l'avance) :** H6, H7, H8.

**H5 — la règle pré-enregistrée ne tranche pas, mais le sens est net.** Le critère demandait une équivalence à ±0,05 : l'IC descend
jusqu'à −0,062. Donc statut **indéterminée**. En revanche `dlind-const` est meilleure sur les 5 seeds et sa différence avec
`dlind-hist` exclut 0 : la version sans clé ni requête n'est **pas moins bonne** que la delta rule du dépôt. C'est une lecture
faite après coup, à signaler comme telle. Contrôle : perturber les requêtes et les clés de `dlind-const` laisse la perte rigoureusement identique
(écart 0,0) : la couche ne fait aucun adressage par contenu.

**Conséquence pour C3.** Le gain de `dlind` sur `smax` (−0,26 avec la φ historique, −0,16 avec la φ correcte) n'a pas besoin d'un adressage
par clés : un oubli exponentiel par tête fait aussi bien ou mieux. Le titre « précision de récupération » n'est pas soutenu
par ces mesures.

**H6 — la φ historique est un régulariseur implicite.** Avec la φ correcte, `dlind` perd 0,095 (val min) et sur-apprend plus (val@1800 : 2,02 → 2,45).
Les conclusions « l'état régularise » (H4, H4b de `HYPOTHESES.md`) sont donc en partie dues à la φ, à ré-attribuer.

**H9 — la valeur est au bord du seuil.** Ratio par seed 0,52 / 0,70 / 0,54 / 0,66 / **−0,13** (s55) ; bootstrap sur 5 seeds (indicatif) [0,14 ; 0,66].
Le contraste (lin-elu1 − hybride-elu1) − 0,5 (lin-elu1 − smax) vaut −0,002, IC [−0,034 ; +0,030] : **ni au-dessus ni au-dessous de 50 %**.
Avec la φ historique, le même calcul (post-hoc, C2 : 80 %) donne 109 %. Le « rattrapage de l'écart » de H1 dépend donc de la φ ; à 5 seeds, je ne peux pas
dire mieux que « l'avantage de l'hybride sur `lin` n'est pas établi avec la φ correcte » : l'IC de l'écart lui-même touche 0 ([−0,003 ; +0,071]).

**Limites à garder en tête.** Corpus de 11 Ko : ces écarts reflètent probablement surtout la régularisation, pas la qualité du mécanisme pour une langue.
`val_entiere_min` est un minimum pris sur la même val que celle rapportée (le pas 1800 donne le même ordre pour H5, H6, H7).
Toutes les variantes sont à 2 couches, d = 96, T = 96.

## 4. Vérifications faites sur ces 45 runs

- 45/45 dossiers complets : 19 points de 0 à 1 800, aucune valeur non finie, `resume.json` cohérent avec `historique.csv`, seeds et variantes corrects.
- Chaque `ENV.json` indique le commit `109319f` et `code_modifie: false` ; les fichiers de code datent d'avant le premier run.
- Différences, moyennes, écarts types et IC recalculés par un autre script (quantile de Student calculé, sans `analyse_c3b.py`) : identiques.
- Reproductibilité : voir § 5.

## 5. Reproductibilité

Rejeu complet (1 800 pas) de `dlind-const` s33 et `dlind-hist` s33, mêmes réglages, 1 thread, sur la même machine : `historique.csv`,
`resume.json` et tous les poids du `checkpoint.npz` sont **identiques octet par octet** (poids : égalité exacte). Les 43 autres runs n'ont pas été rejoués.
Aucune vérification sur une autre machine ou une autre version de NumPy.

## 6. Suite possible

- Le protocole prévoit 8 seeds si un IC ne tranche pas (H5, et H9 au bord). Minimum utile : `dlind-const`, `dlind-hist`,
  `lin-elu1`, `hybride-elu1`, `smax` × 3 seeds (66, 77, 88), soit 15 runs, environ 1 h sur cette machine. À ouvrir par un amendement daté avant les runs.
- Priorité réelle : un corpus de 1 à 5 Mo avec jeu de test séparé (`scripts/preparer_corpus.py`) ; à 11 Ko, aucun modèle n'égale un n-gramme.
- Étiquettes H5/H6 en double avec `HYPOTHESES.md` (C5) : à renommer avant d'y reporter quoi que ce soit.
