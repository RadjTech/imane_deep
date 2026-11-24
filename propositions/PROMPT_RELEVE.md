# Mission : prendre la relève, préparer les données, valider la méthode

Tu travailles sur `imane_deep` : petits modèles de langage français, en NumPy pur, entraînés sur un ordinateur portable sans carte graphique.
Réponds en français. La personne qui te lit n'est pas technicienne : phrases courtes, pas de jargon dans tes messages (le jargon va dans les fichiers).

**Questions auxquelles tu dois répondre à la fin, en une phrase chacune :**
1. Les données sont-elles prêtes et propres ?
2. Notre méthode (`dlind`) fait-elle mieux que l'attention classique (`smax`) sur un texte assez grand ?
3. La version simple (`dlind-const`, sans clé ni requête) suffit-elle ?
4. Le modèle fait-il mieux qu'un simple compteur de lettres (n-gramme) ?

---

## Étape 0 — Prendre la relève (ne modifie rien pendant cette étape)

1. `git pull --ff-only origin main`. Reste sur `main` : **aucune branche, aucun push sans mon accord**.
2. Lis dans l'ordre : `propositions/RELEVE.md`, `propositions/RESULTATS_C3b.md`, `RESEARCH_PROTOCOL.md`, la fin de `RESEARCH_LOG.md` (entrée C3b), `METRICS_CPU.md`.
3. Note ta machine : cœurs physiques, RAM libre, version de Python et de NumPy. Sous Windows, fixe `PYTHONIOENCODING=utf-8`.
4. Lance les garde-fous (quelques secondes chacun) et vérifie qu'ils passent :
   ```
   python3 experiments/hybrid_numpy.py --check-grads
   python3 scripts/gardefou_conformite.py
   python3 scripts/c3b_controles.py --check
   ```
5. Rejoue un résultat connu dans un dossier temporaire (environ 5 min) :
   `python3 scripts/c3b_controles.py --variante dlind-const --seed 33 --out-dir <temporaire>` puis compare `val_entiere_min` à `results/c3b_dlind-const_s33/resume.json` (1,8758).
   Même machine : identique au bit près (vérifié). Autre machine : de petites différences de calcul sont normales ; arrête-toi seulement si l'écart dépasse 0,05.
6. Dis-moi en 5 lignes simples ce que tu as compris, puis passe à l'étape 1 sans attendre. Si un garde-fou échoue : arrête-toi et explique.

À savoir : C1–C5 ont une formule φ fausse pour x ≤ 0 (gardée par défaut pour rester comparable ; `--phi elu1` ou les variantes `-elu1` utilisent la bonne).
Les noms H5 et H6 existent déjà pour C5 : utilise `C7-V1`, `C7-V2`… dans ce cycle. C6 (budget) est déjà réservé : ne le lance pas et n'y touche pas sans que je te le demande.

---

## Étape 1 — Les données

Le corpus actuel fait 11 Ko (4 pages) : trop petit pour conclure. Il faut un corpus français propre, avec train / val / test.

**Livrables** : `data/fr1m/` et `data/fr5m/` (produits par `scripts/preparer_corpus.py`), `data/SOURCES.md`, `scripts/nettoyer_sources.py`, `propositions/RAPPORT_DONNEES.md`.

**Sources**
- Français d'origine, **domaine public** : auteurs morts avant 1910, pas de traductions (le traducteur peut être protégé), pas de textes récents, pas de données personnelles.
- Téléchargement autorisé uniquement depuis gutenberg.org et fr.wikisource.org, en respectant leurs règles d'accès (peu de fichiers, pauses). Pour tout autre site, ou si tu n'as pas internet, demande-moi.
- Au moins 10 œuvres de 8 auteurs différents, aucun auteur au-dessus de 20 % du total, uniquement de la prose (pas de poésie, théâtre, tableaux, listes).
- Garde à part 2 œuvres entières d'auteurs absents du corpus : `data/test_oeuvres.txt`, nettoyées pareil, jamais utilisées pour choisir quoi que ce soit.
- Pour chaque texte, dans `SOURCES.md` : titre, auteur, année de mort, adresse, licence, date, taille, SHA-256. Les textes bruts restent hors dépôt.

**Nettoyage** (`scripts/nettoyer_sources.py`, déterministe, sans accès réseau) : enlever en-têtes et licences, mentions de transcription, numéros de page, notes, titres de chapitre en capitales, `[Illustration]` ;
**recoller les lignes coupées** à l'intérieur d'un paragraphe, garder une ligne vide entre paragraphes ; UTF-8, Unicode NFC. Un fichier `.txt` par œuvre.

**Découpage** (commandes vérifiées sur un texte factice ; sous PowerShell, écris la liste des fichiers à la place de `*.txt`) :
```
python3 scripts/preparer_corpus.py --sources sources_fr/*.txt --nom fr1m --licence "domaine public, voir SOURCES.md" --max-car 1000000 --sortie data
python3 scripts/preparer_corpus.py --sources sources_fr/*.txt --nom fr5m --licence "domaine public, voir SOURCES.md" --max-car 5000000 --sortie data
```
Le script refuse d'écraser un dossier existant : pour corriger, change `--nom`.

**Contrôles à passer, avec les chiffres dans `RAPPORT_DONNEES.md`**
1. `manifest.json` écrit sans erreur ; vocabulaire entre 60 et 160 caractères ; moins de 0,1 % de caractères remplacés par `¤`.
2. Aucune trace de `Project Gutenberg`, `http`, `@`, `***` ; aucune ligne coupée au milieu d'un paragraphe (longueur moyenne d'un paragraphe entre 150 et 1 500 caractères).
3. Tous les caractères de `test.txt` sont dans `corpus.txt`.
4. Doublons : moins de 0,5 % des suites de 8 mots du test existent aussi dans le train.
5. Essai de fumée : `python3 experiments/hybrid_numpy.py --train --types smax,smax --data data/fr1m/corpus.txt --max-steps 10 --log-every 5 --seed 1 --out-dir <temporaire>` finit sans erreur et la perte baisse.
6. Cinq extraits de 300 caractères, pour que je puisse juger le texte moi-même.

Si un contrôle échoue : corrige, ou dis-moi pourquoi ce n'est pas possible. Ne baisse pas un seuil sans l'écrire.

---

## Étape 2 — La référence simple (n-gramme)

`python3 scripts/ngram_baseline.py --data data/fr1m/corpus.txt` : note le résultat, le temps et la mémoire.
Il garde tous les contextes en mémoire : essaie d'abord avec 200 Ko. S'il dépasse 4 Go ou 15 minutes, ne modifie pas ce fichier : écris `scripts/ngram_leger.py`
(même principe, ordre fixé d'avance) et dis clairement ce qui change. C'est la référence à battre : elle est comparée sur la **même val** que les modèles.

---

## Étape 3 — Validation (nouveau cycle C7, ou le premier numéro libre)

**Avant tout calcul**, écris et commite localement `propositions/PROTOCOLE_C7.md` et `scripts/analyse_c7.py` (les critères ne changent plus ensuite ; une modification = un amendement daté, justifié).

- Corpus principal : `data/fr1m`. Confirmation sur `data/fr5m` si la machine le permet.
- Variantes (`scripts/c3b_controles.py`) : `smax`, `lin-hist`, `dlind-hist`, `dlind-elu1`, `dlind-const`.
- Seeds neufs : 101, 202, 303, 404, 505 (5 au minimum). Réglages identiques à C3b : 1 800 pas, 2 couches, d = 96, T = 96, batch 16.
- Métrique principale : perte sur le **test**, une seule évaluation par run, à la fin, avec les poids finaux (aucun choix ne s'appuie sur le test). Métriques secondaires : val (min et pas 1 800), perte sur `test_oeuvres.txt`, vitesse d'entraînement, mémoire.
- Comparaisons **appariées par seed**, IC à 95 % de Student (n−1 degrés de liberté, écart type ddof = 1) :
  - **C7-V1** (notre méthode bat l'attention classique) : IC de (`dlind-elu1` − `smax`) entièrement < 0.
  - **C7-V2** (la version simple suffit) : borne haute de l'IC de (`dlind-const` − `dlind-elu1`) ≤ +0,05.
  - **C7-V3** (la φ d'origine aide-t-elle encore ?) : signe de l'IC de (`dlind-elu1` − `dlind-hist`) ; > 0 = elle aide, < 0 = la bonne formule est meilleure, sinon « on ne sait pas ».
  - **C7-V4** (contre le n-gramme) : sur la même val, le meilleur réseau est-il meilleur que le n-gramme ? (description, pas un critère de succès).
- Un IC qui ne tranche pas donne « indéterminée », et on passe à 8 seeds avant de conclure.

**Script `scripts/evaluer_test.py`** (à écrire, à tester) : relit `checkpoint.npz` et les réglages de la variante, reconstruit le vocabulaire depuis `corpus.txt`, découpe le test en fenêtres non chevauchantes de 96 (chaque caractère prédit une fois, aucun tirage aléatoire),
remplace les caractères inconnus par `¤` (compte-les), écrit `test.json` dans le dossier du run. Contrôle obligatoire : sur la val, après `c3b_controles.installer(...)` de la variante, il doit donner la même valeur que `evaluer_fixe` de `hybrid_numpy.py` (écart < 1e-9).

**Lancer** (priorité basse, 1 thread par processus, au plus un processus par cœur physique) :
```
python3 scripts/lancer_parallele.py --cycle c7a --variantes --configs smax lin-hist dlind-hist dlind-elu1 dlind-const \
    --seeds 101 202 303 404 505 --jobs 2 --data data/fr1m/corpus.txt
```
Le lanceur refuse d'écraser un run terminé et reprend avec `--reprendre`. Pour `fr5m`, utilise `--cycle c7b`.
Mesuré sur la machine actuelle (2 cœurs, 1 thread par run) avec un corpus de 1 Mo : évaluation complète de la val 25 à 45 s par point (19 points par run), pas d'entraînement 0,16 à 0,37 s. Compte **15 à 30 min par run, environ 5 h au total** en deux processus. Fais d'abord un seul run pour confirmer, puis donne-moi la durée estimée.
Sur `fr5m`, l'évaluation actuelle (une fenêtre tous les 16 caractères) devient très longue : ajoute une option `--pas-eval` (défaut 16, comportement inchangé) à un script à part, dis-le dans le protocole, et rejoue le contrôle « identique à C3b » avant.

**Après les runs** : vérifie l'intégrité (tous les runs complets, même commit, code non modifié), recalcule les chiffres avec un deuxième script, puis écris l'entrée dans `RESEARCH_LOG.md`, `HYPOTHESES.md` (étiquettes `C7-…`) et `propositions/RESULTATS_C7.md`.

---

## Règles à ne pas enfreindre

- `main` uniquement, pas de branche, pas de push ni d'effacement d'historique sans mon accord. Commits locaux permis.
- Ne modifie pas les dossiers de `results/` existants, `data/corpus_fr.txt`, les scripts de C1–C5, ni la φ par défaut.
- Protocole commité avant les runs ; un run = un dossier ; pas de résultat « établi » avec moins de 3 seeds ; un résultat négatif se publie comme les autres.
- N'invente aucun chiffre : si tu n'as pas mesuré, écris « non mesuré ». `dlind-const` n'est pas pris en charge par `profiler_inf.py`.
- Avant un calcul long (plus de 30 min), dis-moi combien de temps et de mémoire, et attends mon accord.
- Si tu doutes, demande plutôt que de deviner.

## Compte rendu final attendu (très simple)

1. Les quatre réponses de la première page, en une phrase chacune.
2. Un tableau « entre combien de lettres le modèle hésite » (= exponentielle de la perte) : `smax`, `lin`, `dlind`, `dlind-const`, n-gramme ; plus petit = mieux.
3. Ce qui est sûr, ce qui ne l'est pas, ce qui n'a pas pu être fait.
4. Où sont les fichiers. Puis la question : « veux-tu que je pousse ? ».
