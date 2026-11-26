# Protocole C10 — contexte plus long (T = 256) — PRÉ-ENREGISTRÉ avant tout run (2026-11-25)

> Étape 4 de `PROMPT_SUITE.md`. Commité **avant** le premier run C10 (voir git).
> Toute modification après-coup = amendement daté. Étiquettes `C10-V1`…`C10-V4`.

## 1. Question
L'écart `dlind-elu1 − smax` change-t-il quand le contexte passe de T = 96 (C8) à T = 256,
à **caractères par pas et budget égaux** ? (C9 a montré que le coût relatif bascule en
faveur de la delta rule à long contexte : la qualité suit-elle ?)

## 2. Dispositif

- **Variantes** : `smax`, `dlind-elu1`, `dlind-const` (mêmes définitions que C8,
  `c3b_controles.VARIANTES`).
- **Seeds neufs** : 711, 722, 733 (jamais utilisés ; C8 réserve 661-705 pour son
  amendement éventuel). 3 seeds = minimum absolu : un IC non tranché sera consigné
  « indéterminé », jamais interprété comme une égalité ; amendement possible
  (+744, 755, 766) **avant** toute conclusion.
- **Budget** : 5 400 pas (comme le bras médian de C8) ; **T = 256**, **lot de 6**
  (6 × 256 = 1 536 caractères par pas = 16 × 96 de C8 — le nombre de caractères vus
  par pas est inchangé, seule la longueur du contexte change).
- **Réglages identiques à C8** : 2 couches, d = 96, 4 têtes, AdamW (lr 1e-3, wd 0.05),
  échauffement 100 pas, cosinus réglé sur 5 400, coupure 1.0, corpus `data/fr1m`,
  MIN_COMPTE 20, ¤.
- **Évaluation** : val entière (fenêtres de 256, pas 16 — même protocole que C8 avec sa
  fenêtre) tous les 200 pas ; `checkpoint_best.npz` = val entière minimale ; le **test**
  est évalué **une seule fois** à la fin (poids best ET finaux) — `test_best` principal ;
  `test_oeuvres_v2` une fois. Rien n'est choisi avec le test.
- **Implémentation** : `scripts/c10_controles.py` réutilise la boucle de
  `scripts/c8_github.py` **sans la modifier** (import + surcharge T et lot) — le
  « identique à C8 » est garanti par construction ; `ENV.json` note `T: 256, lot: 6`.
- **Exécution** : locale, 2 processus à 1 thread, priorité basse ; la série saute les
  runs déjà terminés (reprise après coupure) ; un run interrompu est relancé de zéro
  (le re-jeu exact n'est pas garanti après interruption : noté dans le run repris).

## 3. Avertissement métrique (important)

Les pertes à T = 256 et à T = 96 **ne sont pas comparables en valeur absolue** (fenêtres
de contexte différentes ⇒ tâches différentes). Seuls les **écarts appariés entre variantes
évaluées au même T** sont comparés entre eux. C10-V3 est descriptif.

## 4. Critères pré-enregistrés (test_best, apparié par seed, IC 95 % Student, n−1 ddl)

| # | Énoncé | Verdict |
|---|---|---|
| **C10-V1** | IC(`dlind-elu1` − `smax`) @T=256/b5400 : < 0 la delta rule gagne ; > 0 la classique gagne ; 0 dedans → « indéterminé » | IC tranché ou « indéterminé » |
| **C10-V2** | IC(`dlind-const` − `dlind-elu1`) @T=256/b5400, borne haute ≤ +0.05 → la version simple suffit | confirmée / infirmée / indéterminé |
| **C10-V3** (descriptif) | l'écart @T=256 (centre + IC) est-il compatible avec l'écart @T=96/b5400 de C8 (+0.0868, IC [+0.0787 ; +0.0950]) ? — chevauchement des IC | descriptif, pas un critère |
| **C10-V4** (descriptif) | meilleur réseau (val entière) vs n-gramme (1.3402, même val) à T=256 | descriptif |

## 5. Analyse et vérification

- `scripts/analyse_c10.py` → `results/c10_analyse.json` (verdicts + tableau).
- `scripts/verif_c10.py` : recalcul indépendant (quantile de Student par intégration) →
  comparaison IDENTIQUE/ÉCART, comme C7 et C8.
- Intégrité attendue : 9/9 dossiers (resume.json, historique.csv 28 lignes = 0 à 5400 par
  200, checkpoint*.npz, test.json, ENV.json même commit + `code_modifie: false`).

## 6. Règles rappelées

Main uniquement ; push à la fin de l'étape ; ne pas toucher aux results/ existants,
`data/corpus_fr.txt`, `data/fr1m`, fr5m_v2 (déjà conforme), scripts C1-C5 et C8, φ par
défaut ; C6 réservé ; « établi » seulement avec IC tranché et ≥ 3 seeds ; résultats
négatifs publiés comme les autres ; chiffres jamais inventés.

## Amendement n°1 (2026-11-26, commité AVANT les runs complémentaires)

C10-V1 est **indéterminé** à 3 seeds (+0.0022, IC [−0.0412 ; +0.0457]) — conformément au
tableau de décision de la mission (« un IC ne tranche pas → ajoute des seeds »), ajout de
**3 seeds complémentaires : 744, 755, 766**, pour les deux variantes concernées
(`smax`, `dlind-elu1`) uniquement, mêmes réglages (T = 256, lot 6, 5 400 pas). V2 (déjà
tranchée : infirmée) et `dlind-const` ne sont pas concernées — ajouter des seeds à une
comparaison tranchée pour tenter de la retourner serait du shopping de verdict.
Analyse finale : `analyse_c10.py` relancé sur 6 seeds, puis `verif_c10.py`.
Écrit avant le premier run d'amendement.
