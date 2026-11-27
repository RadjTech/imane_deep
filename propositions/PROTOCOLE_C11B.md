# Protocole C11-bis — le bon budget sur les bonnes données (fr5m_v2, 18 000 pas) — PRÉ-ENREGISTRÉ avant tout run (2026-11-26)

> Suite directe de `RESULTATS_C11.md` §3 point 2 : à budget fixe (5 400 pas), plus de
> données n'aide pas le réseau (1,8 époque seulement) alors que le compteur absorbe tout
> (écart réseau↔compteur passé de 0.148 à 0.252). C11-bis test le levier indiqué :
> **fr5m_v2 avec le budget proportionné** (18 000 pas, même budget que le meilleur régime
> de C8). Étiquettes `C11B-V1`…`C11B-V4`.

## 1. Question
Avec les bonnes données ET le bon budget, le réseau rattrape-t-il le compteur de lettres
sur fr5m_v2 (0.252 → ?), et le duel smax / delta rule change-t-il avec le budget comme
sur fr1m (+0.087 à 5 400 → +0.036 à 18 000) ?

## 2. Dispositif (identique à C11, seul le budget change)

- **Corpus** `data/fr5m_v2` (conforme, 0 copie) ; **variantes** `smax`, `dlind-elu1`,
  `dlind-const` ; **seeds neufs** 811, 822, 833 ; **T = 96, lot 16** (1 536 car./pas,
  ~20 époques) ; **18 000 pas**, échauffement 100, cosinus 18 000, AdamW lr 1e-3 wd 0.05.
- **Évaluation** : val entière (fenêtres 96, pas 96) tous les 600 pas ; `checkpoint_best`
  (val min) ; test évalué **une fois** à la fin (best ET final) + `test_oeuvres_v2`.
- **Référence compteur** : n-gramme fr5m_v2 **déjà mesuré avant tout run** (1.222,
  `results/c11_ngram_fr5m.json`, job séparé du cycle C11 — il ne consulte aucun réseau).
- **Exécution** : GitHub Actions (même mécanisme que C8/C11 : 9 ordinateurs en parallèle,
  1 thread, Python 3.13, NumPy 2.3.5, résultats commis sur main par le robot). Local :
  rien (le plafond 72 h est respecté — ~19 h consommées).

## 3. Critères pré-enregistrés (test_best, apparié par seed, IC 95 % Student, n−1 ddl)

| # | Énoncé | Verdict |
|---|---|---|
| **C11B-V1** (descriptif principal) | écart au compteur sur la val : (meilleur réseau − 1.222) à b18000 — comparé à 0.2517 (b5400, même corpus, C11) | descriptif : le budget referme-t-il le trou ? |
| **C11B-V2** | IC(`dlind-elu1` − `smax`) @18k fr5m_v2 : < 0 delta rule gagne ; > 0 classique gagne ; 0 dedans → « indéterminé » | tranché ou indéterminé |
| **C11B-V3** | IC(`dlind-const` − `dlind-elu1`), borne haute ≤ +0.05 → la version simple suffit | confirmée / infirmée |
| **C11B-V4** (descriptif) | test_best de chaque variante à b18000 vs b5400 (seeds différents : moyennes non appariées, comparaison descriptive) | descriptif |

3 seeds (minimum) ; IC non tranché → amendement (+844, 855, 866) avant conclusion.
Les pertes fr1m ↔ fr5m_v2 restent incomparables en absolu (§3 de PROTOCOLE_C11.md).

## 4. Analyse et vérification

`scripts/analyse_c11b.py` → `results/c11b_analyse.json` + `propositions/RESULTATS_C11B_brut.md`
(lancé par le robot) ; `scripts/verif_c11b.py` (quantile recalculé par intégration) →
IDENTIQUE/ÉCART. Intégrité : 9/9 dossiers, historique 31 lignes (0→18000 par 600),
ENV `cycle: c11b`, code non modifié.

## 5. Décision après résultats

- Écart au compteur < 0.05 et V2/V3 tranchées → le compromis qualité/coût de C9/C10 est
  démontré au régime final : rédiger la conclusion du programme (bilan mis à jour).
- Écart au compteur stagné (> 0.15) → le levier suivant est la **capacité** (d = 192,
  `smax` + `dlind-elu1`, 3 seeds, 18 000 pas) — protocole à écrire AVANT de lancer.
- IC indéterminé → amendement seeds.
