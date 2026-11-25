# Protocole C8 (GitHub Actions) — le budget d'entraînement sur `fr1m` — PRÉ-ENREGISTRÉ

> Écrit et commité **avant** tout run C8 (25/11/2026). Toute modification ultérieure = amendement daté, justifié, commité à part.
> Les 25 runs C7 (`e19f051`) ont servi à poser la question et les seuils ; ils ne comptent pas comme confirmation ici.
> Étiquettes locales `C8-V0` à `C8-V5`. C6 n'est pas concerné.

## 1. Question
À 1 800 pas, les 25 runs de C7 finissent tous à leur minimum : les modèles n'ont pas fini d'apprendre. Avec plus d'entraînement,
qui gagne entre `smax` et `dlind`, la version simple `dlind-const` suffit-elle toujours, et l'écart avec le n-gramme se réduit-il ?

Constaté après coup sur C7 (val entière, moyenne de 5 seeds, **exploratoire**) : `dlind-elu1` est devant `smax` jusqu'au pas 1 250 environ
(−0,17 au pas 400), puis `smax` la dépasse vers 1 300 et l'écart grandit (+0,022 au pas 1 800). C8 mesure la suite.

## 2. Hypothèses et critères (fixés à l'avance)

Comparaisons appariées par seed ; IC 95 % de Student (n−1 ddl, écart type ddof = 1) ; métrique principale = perte **test** de `data/fr1m`
avec les poids retenus par la val entière (`test_best`). Une différence négative veut dire que la première variante est meilleure.

| # | Énoncé | Confirmée si |
|---|---|---|
| C8-V0 | Réplication de C7-V1 sur 5 seeds neufs, à 1 800 pas : `smax` est devant `dlind-elu1` | IC de (`dlind-elu1` − `smax`) entièrement > 0 |
| C8-V1 | `dlind-elu1` bat `smax` à 18 000 pas | IC de (`dlind-elu1` − `smax`) entièrement < 0 |
| C8-V2 | `dlind-elu1` bat `smax` à 5 400 pas | IC de (`dlind-elu1` − `smax`) entièrement < 0 |
| C8-V3 | `dlind-const` suffit à 18 000 pas | borne haute de l'IC de (`dlind-const` − `dlind-elu1`) ≤ +0,05 |
| C8-V4 | L'écart `dlind-elu1` − `smax` évolue entre 1 800 et 18 000 pas | IC de la différence d'écarts (appariée) ne contient pas 0 ; le signe dit qui en profite |
| C8-V5 | (description, pas un critère) meilleur réseau contre n-gramme (1,3402, val entière) | — |

Si un IC ne tranche pas (« indéterminée »), ajouter les seeds 661, 672, 683, 694, 705 par un amendement daté, avant de conclure.
Un IC entièrement du mauvais côté donne « infirmée ».

## 3. Dispositif
- **Variantes** : `smax`, `dlind-elu1` (φ ELU+1), `dlind-const` (sans clé ni requête). `lin-hist` et `dlind-hist` sont laissées de côté (C7).
- **Seeds neufs** (jamais utilisés) : 611, 622, 633, 644, 655.
- **Budgets** : 1 800, 5 400 et 18 000 pas ; le cosinus est réglé sur chaque budget. Soit 3 × 3 × 5 = **45 runs**.
- **Réglages de C7** : 2 couches, d = 96, 4 têtes, T = 96, lot de 16, AdamW (lr 1e-3, wd 0,05), échauffement 100 pas, coupure 1,0, corpus `data/fr1m`.
- **Évaluation** : val entière (fenêtres tous les 16 caractères, comme C7) tous les 100 pas (1 800), 200 (5 400) ou 600 (18 000).
- **Meilleurs poids** : ceux de la val entière minimale. Le **test** est évalué une seule fois par jeu de poids (finaux et meilleurs), à la fin du run.
  Aucune décision ne s'appuie sur le test.
- **Secondaires** : poids finaux, `data/test_oeuvres_v2.txt` (œuvres jamais vues, nettoyées comme le corpus), temps par point.
- **Logiciel** : Python 3.13, NumPy 2.3.5, un seul thread de calcul, un ordinateur GitHub par run.

## 4. Garde-fous
- `--check-grads`, `gardefou_conformite.py`, `c3b_controles.py --check` passent.
- `scripts/c8_github.py --controle-c7` refait les 300 premiers pas du run C7 seed 101 de chaque variante ; écart max toléré 1e-4
  (val, train, val entière). Cela prouve que le nouveau programme est la même expérience.
- L'analyse `scripts/analyse_c8_github.py` a été testée sur des résultats fabriqués dont la réponse est connue.
- Les verdicts sont refusés tant que les 45 runs ne sont pas tous là.

## 5. Données
- `data/fr1m` inchangé (C7 ; 0 copie de 60 caractères entre train, val et test).
- `data/test_oeuvres_v2.txt` = `normaliser` (de `scripts/preparer_corpus.py`) appliqué à `data/test_oeuvres.txt`
  (SHA-256 `d9911b986cd1a5de5961945fd4d551371d09bbc7e2a389937e1832620f959a11`, 1 234 422 caractères, 140 inconnus).
  L'ancien fichier, qui gardait 13 154 apostrophes courbes, reste en place et n'est plus cité.

## 6. Lancement
`.github/workflows/c8.yml`, déclenchement manuel : d'abord **petit-essai**, puis **complet**. La liste des essais vient de
`scripts/c8_matrice.py` (pas d'entrée libre). Les résultats reviennent dans `results/c8g_*` et `propositions/RESULTATS_C8_brut.md`
(commit fait par le robot GitHub, mode complet seulement).

## 7. Limites connues
- Les ordinateurs GitHub peuvent différer d'un run à l'autre : la perte peut bouger un peu (bien moins que les écarts mesurés), les temps ne sont pas comparables.
- Ce protocole ne mesure ni la vitesse de génération ni la mémoire (C9), ni un contexte plus long (C10).
- Corpus de 1 Mo : avec 18 000 pas (environ 32 passages), le sur-apprentissage est attendu ; c'est la raison du choix des poids par la val.
