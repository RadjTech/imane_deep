# Protocole C7 — validation sur corpus réel (fr1m) — PRÉ-ENREGISTRÉ avant tout calcul

**Date** : 2026-11-24 · **Rédigé et commité avant le premier run C7** (voir historique git).
Toute modification après-coup = un amendement daté, justifié, commité séparément.

## 1. Question
Les conclusions de C3b (établies sur un corpus de 11 Ko où un n-gramme 1.683 bat tout le
monde) tiennent-elles sur un corpus réel de ~1 Mo de prose française du domaine public
(`data/fr1m`) ? Concrètement : (1) `dlind` bat-il `smax` ? (2) la version simple
`dlind-const` (sans clé ni requête) suffit-elle ? (3) la φ historique (bug d'origine)
aide-t-elle encore ? (4) le meilleur réseau fait-il mieux que le n-gramme ?

## 2. Critères pré-enregistrés (verdicts, comparaisons appariées par seed)

IC = intervalle de confiance de Student à 95 % (quantile t à n−1 ddl, écart type ddof = 1,
différences appariées par seed).

| # | comparaison | critère de confirmation |
|---|---|---|
| **C7-V1** | `dlind-elu1` − `smax` (perte **test**) | IC entièrement < 0 → confirmée |
| **C7-V2** | `dlind-const` − `dlind-elu1` (perte **test**) | borne haute de l'IC ≤ **+0.05** → confirmée |
| **C7-V3** | `dlind-elu1` − `dlind-hist` (perte **test**) | signe de l'IC : > 0 → « la φ historique aide encore » ; < 0 → « la φ corrigée est meilleure » ; 0 dans l'IC → « on ne sait pas » |
| **C7-V4** | descriptif, pas un critère | meilleur réseau (val entière) vs n-gramme Kneser-Ney (`results/c7_ngram_fr1m.json`, même val, même protocole d'évaluation) |

**Règle d'escalade** : un IC qui ne tranche pas → verdict « indéterminée » ; le protocole
de C3b prévoit alors le passage à 8 seeds (ajouter 111, 222, 333, 444 par amendement daté)
**avant** toute conclusion.

## 3. Dispositif

- **Variantes** (`scripts/c3b_controles.py`, dict `VARIANTES`) : `smax`, `lin-hist`,
  `dlind-hist`, `dlind-elu1`, `dlind-const`. Aucune autre. C6 (budget) n'est pas concerné
  et n'est pas lancé.
- **Seeds neufs** : 101, 202, 303, 404, 505 (5 au minimum).
- **Réglages identiques à C3b** : 1 800 pas, lr 1e-3, warmup 100, 2 couches, d = 96,
  H = 4, T = 96, batch 16 — 1 800 pas × batch × T constant pour toutes les variantes.
- **Corpus principal** : `data/fr1m/corpus.txt` (998 538 car. ; train 90 % / val 10 %
  gérés par le script d'entraînement ; test séparé `data/fr1m/test.txt` jamais vu pendant
  l'entraînement). **Confirmation** sur `data/fr5m` (cycle `c7b`) si la machine le permet —
  subject à l'accord préalable pour la durée (cf. règles).
- **Lancement** : `python3 scripts/lancer_parallele.py --cycle c7a --variantes --configs
  smax lin-hist dlind-hist dlind-elu1 dlind-const --seeds 101 202 303 404 505 --jobs 2
  --data data/fr1m/corpus.txt` (1 thread par processus, un processus par cœur physique ;
  le lanceur trace ENV.json avec le commit et refuse d'écraser un run terminé).
- **Run de calibration** : le premier run (`dlind-const`, seed 101) est lancé seul pour
  mesurer la durée réelle par run et donner l'estimation totale ; il compte comme un run
  ordinaire du dispositif (protocole commité avant). Les 24 autres attendent l'accord
  explicite sur la durée estimée.

## 4. Métriques

- **Principale** : perte sur le **test** (`data/fr1m/test.txt`), **une seule évaluation**
  par run, **à la fin**, sur les **poids finaux** (aucun choix ne s'appuie sur le test ;
  le test n'est évalué qu'après la fin de tous les runs).
- **Secondaires** : val (min et pas 1800, `resume.json`), perte sur
  `data/test_oeuvres.txt` (2 œuvres entières d'auteurs hors corpus), vitesse
  d'entraînement (déduite de `historique.csv`), mémoire d'évaluation (RSS du test).
- **Évaluateur test** : `scripts/evaluer_test.py` — relit `checkpoint.npz` + `resume.json`
  (variante réinstallée via `c3b_controles.installer`), reconstruit le vocabulaire depuis
  `corpus.txt`, découpe en fenêtres **non chevauchantes** de 96 (chaque caractère prédit
  une fois, aucun tirage aléatoire), remplace les caractères hors train (compte < 20,
  comme `preparer_corpus.py`) par `¤` et les compte, écrit `test.json` dans le dossier du run.
  **Contrôle obligatoire avant usage** : sur la val du corpus 11 Ko, il doit reproduire
  `evaluer_fixe` de `hybrid_numpy.py` à **< 1e-9** près (checkpoint `dlind-const` s33).
- n-gramme : `scripts/ngram_baseline.py --data data/fr1m/corpus.txt` →
  `results/c7_ngram_fr1m.json` (mesuré : 1.3402 sur la val entière — ce chiffre est
  **hors verdict**, il sert au descriptif C7-V4 ; mesuré avant les runs, sans consultation
  du test).

## 5. Après les runs

1. Intégrité : 25/25 dossiers complets (resume.json, historique.csv 19 lignes,
   checkpoint.npz, ENV.json même commit, `code_modifie: false`).
2. Recalcul indépendant par un deuxième script (`scripts/c7_verification.py`, commis
   avant les runs aussi) : comparaison IDENTIQUE/ÉCART avec `analyse_c7.py`.
3. Écriture de `propositions/RESULTATS_C7.md`, entrée `RESEARCH_LOG.md` (étiquettes
   `C7-V…` — H5/H6 sont déjà pris par C5 et C3b), statuts dans `HYPOTHESES.md`.
4. **Aucun push sans accord explicite.**

## 6. Règles rappelées

Pas de branche ; pas de push sans accord ; ne pas toucher à `results/` existants,
à `data/corpus_fr.txt`, aux scripts C1–C5, à la φ par défaut ; protocole avant les runs ;
un run = un dossier ; pas de « établi » à moins de 3 seeds ; un résultat négatif se
publie comme les autres ; aucun chiffre inventé (« non mesuré » sinon).
