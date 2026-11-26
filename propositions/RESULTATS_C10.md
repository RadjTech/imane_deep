# Résultats C10 — contexte plus long (T = 256) (2026-11-25)

> Protocole `PROTOCOLE_C10.md` commité avant tout run (`74b0af0`). 9 runs locaux (3
> variantes × seeds 711/722/733), 5 400 pas, T = 256, lot 6 (1 536 caractères/pas —
> identique à C8), 2 processus à 1 thread, ~2 h 06 par run, 9/9 intègres (ENV `cycle: c10`,
> `T: 256`, code non modifié ; historique 29 lignes = en-tête + 28 points — le protocole
> §5 disait « 28 » sans compter l'en-tête, erreur de comptage bénigne).
> Vérification indépendante : **IDENTIQUE** (4.9e-05) — `scripts/verif_c10.py`.

## 1. Tableau (test_best — poids choisis par la val, une seule éval du test)

| Variante | test_best (3 seeds) | moyenne ± écart type | val entière | œuvres_v2 |
|---|---|---|---|---|
| `smax` | 1.6257 1.6370 1.6069 | **1.6232 ± 0.0152** | 1.6209 | 1.7292 |
| `dlind-elu1` | 1.6159 1.6313 1.6292 | 1.6255 ± 0.0084 | 1.6227 | **1.7263** |
| `dlind-const` | 1.6574 1.6595 1.6556 | 1.6575 ± 0.0020 | 1.6586 | 1.7619 |

Tous les runs finissent à leur minimum (pas_best = 5 400, comme C8-b5400 : encore du gain
disponible).

## 2. Verdicts pré-enregistrés

| # | Comparaison | Différence | IC 95 % | Verdict |
|---|---|---|---|---|
| **C10-V1** | dlind-elu1 − smax @T=256 | +0.0022 | [−0.0412 ; +0.0457] | **indéterminé** (0 dans l'IC — jamais interprété comme une égalité) |
| **C10-V2** | dlind-const − dlind-elu1 ≤ +0.05 | +0.0320 | [+0.0115 ; +0.0525] | **infirmée** (borne haute +0.0525 > +0.05, de justesse) |
| **C10-V3** (descriptif) | IC T=256 vs écart C8 T=96 [+0.0787 ; +0.0950] | — | **IC disjoints** | **l'écart a changé avec le contexte** : l'avantage de smax présent à T=96 n'est plus détecté à T=256 |
| **C10-V4** (descriptif) | val — meilleur réseau vs n-gramme | smax 1.6209 vs 1.3402 | écart +0.2807 | le réseau T=256 est loin derrière le compteur **à ce budget** (voir note) |

**Amendement n°1 (daté, commité avant les runs complémentaires)** : V1 étant indéterminé,
ajout des seeds 744, 755, 766 pour `smax` et `dlind-elu1` (6 runs, ~6 h) — uniquement pour
V1 ; V2 reste à 3 seeds (ajouter des seeds à une comparaison déjà tranchée pour tenter de
la retourner serait du « shopping de verdict », refusé). La conclusion définitive de C10
sera écrite après ces runs (même protocole, même test, une éval).

## 3. Lecture honnête

1. **Le résultat central** : à T = 96 et budget égal, smax battait dlind-elu1 de +0.0868
   (IC sans 0, C8) ; à T = 256, la différence tombe à +0.0022 avec un IC contenant 0 **et
   disjoint de l'IC de C8**. Le désavantage de qualité de la delta rule **disparaît quand
   le contexte grandit** — exactement là où C9 montre qu'elle est 4-6× plus rapide et
   ÷100-7 000 en mémoire. Le compromis qualité/coût se déplace en sa faveur à long
   contexte. V1 reste officiellement « indéterminé » tant que les seeds complémentaires
   n'ont pas resserré l'IC.
2. **dlind-const à T=256** : +0.0320 derrière dlind-elu1 (borne +0.0525, de justesse au-delà
   du seuil) — la version simple ne suffit pas pleinement à ce contexte non plus, même si
   son écart est 2.7× plus petit qu'à T=96/b5400 (+0.0868).
3. **V4 (note de prudence)** : les pertes à T=256 ne se comparent pas aux pertes à T=96
   (tâche différente, §3 du protocole). L'écart au n-gramme (+0.28) dit que le modèle
   T=256 à 5 400 pas est encore peu entraîné à **exploiter** les 256 caractères de
   contexte — pas que le contexte nuit. Le n-gramme, lui, n'a pas de « contexte long » à
   apprendre : 0.006 nat (C8) reste la meilleure position du réseau à ce jour.
4. **Contre-hypothèses** : mêmes batchs par caractère que C8 (1 536/pas), même
   protocole d'éval entre variantes, test lu une fois, seeds appariés ; le hasard des
   3 seeds est la limite principale (d'où l'amendement).
5. **Coût du contexte** : T=256 coûte ~2.3× plus de temps par run que T=96 à caractères
   égaux (attention O(T²) : 256²/96² = 7.1× sur le bloc d'attention, dilué par le reste).

## 4. Ce qui manque / suite

- Les 6 runs d'amendement (744/755/766, smax + dlind-elu1) → conclusion définitive de V1.
- C11 (fr5m_v2) lancé **sur GitHub Actions** par l'utilisateur (amendement n°1 de
  PROTOCOLE_C11.md) — la série locale `c11_serie.sh` n'est pas lancée ; vérification et
  `RESULTATS_C11.md` dès l'arrivée de `results/c11_analyse.json`.
- Bilan final (étape 6) après C10-amendement et C11.
