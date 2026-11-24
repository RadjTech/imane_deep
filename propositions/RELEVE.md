# Relève — 24/11/2026 (mise à jour après C3b)

Passage de relais. **Ordre de lecture** : ce fichier, puis `RESULTATS_C3b.md` (résultats), puis `PROMPT_RELEVE.md` (mission suivante).
« Mesuré » = fait avec les scripts de ce dossier ou de `scripts/`.
Les chiffres d'un audit fait par un sous-agent sont dans `AUDIT.md`, marqués comme tels quand je ne les ai pas revérifiés.

## 0. Où on en est (une minute)

- **C1–C5** : faits, sur un corpus de 11 Ko. **C6 (budget)** : cycle du collaborateur, pas lancé ici, à ne pas toucher.
- **C3b** (re-mesure des fondations) : **terminé**, 45 runs, seeds neufs. Verdicts (étiquettes locales `C3b-H…`, car H5/H6 de `HYPOTHESES.md` sont celles de C5) :
  `dlind` reste devant `smax` avec la φ correcte (−0,165, 5/5) ; la φ historique aidait `dlind` (+0,095) ; normaliser les clés ne suffit pas ;
  la version sans clé ni requête (`dlind-const`) fait au moins aussi bien que `dlind` (−0,043, 5/5 ; verdict pré-enregistré « indéterminée ») ;
  l'hybride avec φ correcte rattrape 47 % de l'écart (seuil 50 %, dans le bruit).
- **Verrou** : le corpus de 11 Ko. Un n-gramme (1,683) fait mieux que tous les modèles ; à cette taille on mesure surtout la régularisation.
- **Suite** : données puis validation sur un corpus plus grand → `PROMPT_RELEVE.md`.
- **Règles** : travail sur `main`, aucune branche ; pas de push sans demande ; ne jamais réécrire un dossier de résultats ; protocole commité avant les runs.

## 1. Ce qui vient d'être fusionné dans `main` (commit `2536e85`)

- **φ à deux versions dans `experiments/hybrid_numpy.py`.** Le défaut reste « historique » (C1–C5) ; `--phi elu1` active la version
  corrigée. Mesuré : avec les options par défaut, l'entraînement est identique à `main` au bit près (`dlind,dlind`, `lin,smax`, `smax,smax`).
  Donc C1–C5 restent reproductibles tels quels.
- **Évaluation fixe** sur toute la val (1 117 prédictions, sans tirage aléatoire), `checkpoint_best.npz`, champs `phi` et `val_fixe_*`
  dans `resume.json`, 4e colonne `perte_val_fixe` dans `historique.csv`.
- `profiler_inf.py --phi`, à préciser pour un checkpoint entraîné avec `elu1`.
- `scripts/` : `lancer_parallele.py` (2 processus à 1 thread sur 2 cœurs), `agreger.py` (moyennes et IC appariés, écart type ddof = 1),
  `gardefou_conformite.py`, `ngram_baseline.py`, `c3b_controles.py`, `preparer_corpus.py`.

## 2. Le bug φ (mesuré)

La `phi` de C1–C5 vaut `exp(x)+1` pour x ≤ 0 au lieu de `exp(x)` : elle saute de 2 à 1 en zéro et vaut au moins 1 partout. La dérivée est
juste par morceaux, donc les différences finies (`--check-grads`) ne le voient pas ; `gardefou_conformite.py` le voit et le teste
(avec `--phi elu1` : conforme). À l'initialisation, les clés normalisées de `dlind` sont presque parallèles (cosinus 0,95) : l'écriture
« remplaçante » ressemble à une moyenne exponentielle.

Pilotes (1 seed ⇒ **indication**, sauf mention) :

| Variante | seed | val_min | val @1800 |
|---|---|---|---|
| `dlind`, φ historique (C3) | 1337 | 1,8931 | 1,9932 |
| `dlind`, φ corrigée | 1337 | 1,9718 | 2,5472 |
| `dlind` **sans clé ni requête** (oubli exponentiel pur) | 1337 | **1,8699** | **1,9169** |
| `dlind`, φ historique (C3) | 2024 | 1,8873 | 2,0075 |
| `dlind` sans clé ni requête | 2024 | 1,8556 | 1,9067 |
| `lin`, φ historique (C2) | 1337 | 2,2349 | 2,8330 |
| `lin`, φ corrigée | 1337 | 2,2039 | 3,2332 |
| `lin`, φ historique + clés normalisées | 1337 | 2,2471 | 2,5455 |
| `smax` (C2) | 1337 | 2,1235 | 3,1385 |

**Depuis, C3b a vérifié sur 5 seeds neufs** (`RESULTATS_C3b.md`) : (a), (b) et (c) ci-dessous vont dans le même sens, (b) est confirmée pour `dlind` (H6).

Lecture des pilotes (1 à 2 seeds, avant C3b) : (a) une grande partie du gain de `dlind` vient d'un oubli exponentiel, pas de
l'adressage par clés (la version sans clé fait aussi bien ou mieux, 2 seeds) ; (b) l'effet « l'état régularise » (H4, H4b) semble porté
par la φ historique : avec la φ corrigée, `lin` sur-apprend comme `smax` (3,23 contre 3,14) et `dlind` passe de 1,99 à 2,55 au pas 1800 ;
(c) à ce seed, `dlind` reste devant `smax` avec la φ corrigée (1,972 contre 2,124).

**Référence à citer** : un n-gramme Kneser-Ney (`scripts/ngram_baseline.py`, entraîné en 0,3 s) fait **1,683** sur la même val (perte moyenne, 1 117 prédictions, fenêtres de 96).
C'est meilleur que tous les modèles du dépôt, `dlind`×4 de C5 compris (1,8441). À 11 Ko, les cycles mesurent surtout la régularisation,
pas la qualité du mécanisme : le corpus est le premier verrou.

## 3. À regarder sur C4 et C5 (constats d'audit ; G5 revérifié par moi, le reste non)

- **C4** : Δ_A (val_min, éval tirée) et Δ_B (checkpoint au pas 1800, éval fixe) ne sont pas pris au même point. Le verdict H3 ne change pas
  (Δ_B ≈ +2), mais l'écart ne se lit pas directement. `analyse_c4_amende.py` propose une version alignée, à utiliser seulement si C4 est relancé.
- **Écarts types** : C4 et C5 utilisent ddof = 0 (ex. 0,0122 pour `dlind`×4), C2 et C3 ddof = 1 (0,0149 avec ddof = 1).
- **`results/profile_inf_lin-lin.json`** contient un profil de checkpoint `dlind` (`/tmp/smoke_dlind`, 1 552 tok/s @4096), pas celui de
  `lin,lin` publié (2 693). À refaire avant de citer « lin,lin plat à 2 693 ».
- H1 « confirmée » alors que H1b est infirmée : l'audit dit qu'elle devait retomber à « indication » (non revérifié).
- README, section « Contexte » : « 16-32× de mémoire » alors que C4 mesure ×7,3 (8 % des paramètres restent en float32).

## 4. Pour continuer

1. **Données puis validation** : la mission complète est dans `PROMPT_RELEVE.md` (corpus `fr1m` / `fr5m`, protocole, règles, commandes testées).
2. **Passer à 8 seeds pour H5 et H9 de C3b** (prévu par le protocole, non lancé) : 15 runs, environ 1 h sur la machine locale.
   `python scripts/lancer_parallele.py --cycle c3b --variantes --configs dlind-const dlind-hist lin-elu1 hybride-elu1 smax --seeds 66 77 88`, à ouvrir par un amendement daté.
   `analyse_c3b.py` ne compte que les seeds 11 à 55 : il faut une version étendue, pré-enregistrée, avant ces runs.
3. **Si le prochain cycle reprend φ historique**, rien à changer : le défaut est le même que C1–C5. **Pour passer à la φ corrigée**, re-mesurer les témoins dans le même lot.
4. CI en matrice : `propositions/cycle.yml` (à placer dans `.github/workflows/`), déclenchement manuel, un job par configuration × seed. Syntaxe vérifiée, jamais exécutée sur GitHub.

## 5. Non vérifié

- Le sous-agent de veille bibliographique (conditions GitHub Actions, sources de corpus français) et le moteur NumPy rapide ont été
  interrompus (limite d'API, blocage) : leur travail n'est ni vérifié ni inclus ici.
- Le pilote `dlind` φ corrigée, seed 2024, a été arrêté avant la fin (ma priorité était C4).
- C3b : deux runs rejoués à l'identique, les 43 autres non ; aucune vérification sur une autre machine.
- Les étiquettes H5/H6 sont en double entre C3b et C5 (voir `HYPOTHESES.md`) : à clarifier avant d'y recopier des verdicts.

## 6. Fichiers de ce dossier

`RESULTATS_C3b.md` (résultats et lecture de C3b) · `PROMPT_RELEVE.md` (mission suivante) · `AUDIT.md` (rapport d'audit, 2 pages) · `PROTOCOLE_C3b.md` (protocole pré-enregistré, exécuté) · `analyse_c4_amende.py` (analyse C4 alignée, non exécutée sur les données publiées) · `cycle.yml` (CI en matrice) · `pilotes/` (CSV et resume.json des 5 pilotes complets).
