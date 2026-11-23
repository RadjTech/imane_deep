# Relève — 24/11/2026

Passage de relais. « Mesuré » = fait aujourd'hui avec les scripts de ce dossier ou de `scripts/`.
Les chiffres d'un audit fait par un sous-agent sont dans `AUDIT.md`, marqués comme tels quand je ne les ai pas revérifiés.

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

Lecture possible, à confirmer sur des seeds neufs : (a) une grande partie du gain de `dlind` vient d'un oubli exponentiel, pas de
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

1. **Si le prochain cycle reprend φ historique**, rien à changer : le défaut est le même que C1–C5.
2. **Pour passer à la φ corrigée**, il faut re-mesurer les témoins dans le même lot (le protocole `PROTOCOLE_C3b.md` le fait, seeds neufs).
   `python scripts/lancer_parallele.py --cycle c3b --variantes --configs smax hybride-hist hybride-elu1 lin-hist lin-elu1 lin-hist-nk dlind-hist dlind-elu1 dlind-const --seeds 11 22 33 44 55`
   puis `python scripts/agreger.py --prefixes c3b --ref dlind-hist --metrique val_entiere_min`.
3. **Un corpus de 1 à 5 Mo** avec train / val / test (`scripts/preparer_corpus.py`, texte local déjà sous la main).
   Rien n'est téléchargé par ce script.
4. CI en matrice : `propositions/cycle.yml` (à placer dans `.github/workflows/`), déclenchement manuel, un job par configuration × seed. Syntaxe vérifiée, jamais exécutée sur GitHub.

## 5. Non vérifié

- Le sous-agent de veille bibliographique (conditions GitHub Actions, sources de corpus français) et le moteur NumPy rapide ont été
  interrompus (limite d'API, blocage) : leur travail n'est ni vérifié ni inclus ici.
- Le pilote `dlind` φ corrigée, seed 2024, a été arrêté avant la fin (ma priorité était C4).
- Les conclusions de la section 2 ne reposent que sur 1 à 2 seeds ; `PROTOCOLE_C3b.md` propose 5 seeds neufs.

## 6. Fichiers de ce dossier

`AUDIT.md` (rapport d'audit, 2 pages) · `PROTOCOLE_C3b.md` (brouillon, à valider) · `analyse_c4_amende.py` (analyse C4 alignée, non exécutée sur les données publiées) · `cycle.yml` (CI en matrice) · `pilotes/` (CSV et resume.json des 5 pilotes complets).
