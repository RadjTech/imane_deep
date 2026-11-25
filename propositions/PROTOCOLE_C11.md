# Protocole C11 — plus de données (fr5m_v2, 5.0 M caractères) — PRÉ-ENREGISTRÉ avant tout run (2026-11-25)

> Suite indiquée par le tableau de décision de `PROMPT_SUITE.md` (écart au n-gramme passé
> sous 0.3 nat en C8). Commité **avant** le premier run C11 et avant la mesure du
> n-gramme fr5m_v2. Étiquettes `C11-V1`…`C11-V4`.

## 1. Question
Avec 5× plus de texte propre (fr5m_v2, 14 œuvres / 10 auteurs, **0 copie de 60 car.**),
qui gagne entre `smax` et la delta rule à budget égal, et l'écart au compteur de lettres
continue-t-il de se réduire ? (C8 : meilleur réseau à 0.0064 nat du n-gramme sur fr1m.)

## 2. Dispositif

- **Corpus** : `data/fr5m_v2` (train 90 % / val 10 % / test.txt séparé — manifest vérifié
  par `scripts/verifier_corpus.py`, conforme). **L'ancien fr5m (défaut 2) n'est pas utilisé.**
- **Variantes** : `smax`, `dlind-elu1`, `dlind-const`. **Seeds** : 711, 722, 733 — **les
  mêmes que C10** (l'init et l'ordre des batchs sont appariés entre corpus ; les pertes
  restent non comparables en absolu, les écarts relatifs le sont).
- **Budget et réglages** : 5 400 pas, **T = 96, lot 16 (1 536 car./pas — identique à C8)** ;
  seules les données changent par rapport à C8-b5400. AdamW lr 1e-3 wd 0.05, warmup 100,
  cosinus 5 400, coupure 1.0, MIN_COMPTE 20.
- **Évaluation** : val entière (fenêtres de 96, **pas 96** — la val de fr5m_v2 fait 474 K
  caractères, un pas de 16 coûterait 6× plus pour un verdict identique) tous les 200 pas ;
  `checkpoint_best` (val min) ; test évalué **une seule fois** à la fin (best ET final) ;
  `test_oeuvres_v2` une fois. Rien n'est choisi avec le test.
- **Référence compteur** : n-gramme Kneser-Ney (`scripts/ngram_baseline.py --data
  data/fr5m_v2/corpus.txt --out results/c11_ngram_fr5m.json`) mesuré sur la **val entière**
  de fr5m_v2 **avant** les runs C11 (5-60 s, hors réseau).
- **Exécution** : locale, 2 processus à 1 thread, priorité basse, après la fin de C10
  (règle : un processus par cœur) ; la série saute les runs terminés (reprise) ; run
  interrompu = relancé de zéro et noté.
- **Implémentation** : `scripts/c11_controles.py` réutilise la boucle de
  `scripts/c8_github.py` sans modification (défauts T=96/lot=16 inchangés, `--pas-eval 96`,
  `--data data/fr5m_v2/corpus.txt`) ; ENV note `cycle: c11`.

## 3. Avertissement métrique

Les pertes sur fr1m et fr5m_v2 ne sont **pas comparables en absolu** (tests différents).
Les comparaisons entre corpus portent sur les **écarts relatifs** (différences appariées
entre variantes du même corpus, écart réseau−compteur du même corpus), jamais sur des
pertes brutes.

## 4. Critères pré-enregistrés (test_best, apparié par seed, IC 95 % Student, n−1 ddl)

| # | Énoncé | Verdict |
|---|---|---|
| **C11-V1** | IC(`dlind-elu1` − `smax`) @fr5m_v2/b5400 : < 0 delta rule gagne ; > 0 classique gagne ; 0 dedans → « indéterminé » | IC tranché ou « indéterminé » |
| **C11-V2** | IC(`dlind-const` − `dlind-elu1`), borne haute ≤ +0.05 → la version simple suffit | confirmée / infirmée / indéterminé |
| **C11-V3** (descriptif) | meilleur réseau (val fr5m_v2) vs n-gramme fr5m_v2 (val) — écart nat | descriptif |
| **C11-V4** (descriptif) | l'écart (`dlind-elu1` − `smax`) @fr5m_v2 est-il compatible avec celui @fr1m/b5400 de C8 (+0.0868 [+0.0787 ; +0.0950]) ? chevauchement des IC | descriptif |

3 seeds = minimum : tout IC non tranché est consigné « indéterminé » (amendement possible :
+744, 755, 766 avant conclusion).

## 5. Analyse et vérification

`scripts/analyse_c11.py` → `results/c11_analyse.json` ; `scripts/verif_c11.py` (quantile
recalculé par intégration) → IDENTIQUE/ÉCART. Intégrité attendue : 9/9 dossiers + JSON du
n-gramme ; historique.csv 28 lignes (0 à 5400 par 200).

## 6. Règles rappelées

Main uniquement ; protocole avant les runs ; push en fin d'étape ; ne pas toucher aux
results/ existants ni aux corpus déjà validés ; φ par défaut inchangée ; C6 réservé ;
« établi » seulement avec IC tranché et ≥ 3 seeds ; résultats négatifs consignés.
