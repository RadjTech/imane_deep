# Mission autonome : réparer, mesurer, conclure

Tu reprends `imane_deep` en **autonomie complète**. Tu n'as pas besoin de ma confirmation, même pour des calculs de plusieurs heures : lance tout ce qu'il faut, dans la bonne direction, jusqu'au bilan final.
Je ne suis pas technicien : tes comptes rendus sont en français simple, le détail va dans les fichiers.
Cette mission remplace la règle « attends mon accord » de `PROMPT_RELEVE.md` ; ses autres règles restent valables.

## Pour commencer
1. `git pull --ff-only origin main`, reste sur `main`.
2. Relis `propositions/RELEVE.md`, `propositions/RESULTATS_C7.md`, `RESEARCH_PROTOCOL.md`.
3. Lance les garde-fous : `python3 experiments/hybrid_numpy.py --check-grads`, `python3 scripts/gardefou_conformite.py`, `python3 scripts/c3b_controles.py --check`.

## Ce que tu peux faire sans me demander
- Lancer tous les calculs nécessaires, écrire scripts, protocoles et analyses, commiter sur `main`.
- Pousser sur `main` à la fin de chaque étape : `git pull --rebase origin main` d'abord (le robot GitHub ou moi pouvons pousser entre-temps), jamais `--force`, **aucune branche**.
- Télécharger uniquement depuis fr.wikisource.org et gutenberg.org (domaine public, auteurs morts avant 1910, pas de traductions).
- Choisir toi-même la suite avec le tableau de décision ci-dessous.

## Ce que tu ne fais jamais
- Dépenser de l'argent, créer un compte, utiliser un service payant, envoyer des données ailleurs, sortir du dépôt et de ton dossier de travail.
- Effacer ou réécrire un résultat existant, modifier `data/corpus_fr.txt`, `data/fr1m`, les dossiers `results/` existants, les scripts de C1–C5, la φ par défaut. C6 reste réservé.
- Changer un critère après avoir vu les résultats, choisir quoi que ce soit avec le texte de test, annoncer « établi » avec moins de 3 seeds, inventer un chiffre.

## Quand t'arrêter (et seulement là)
Un garde-fou échoue deux fois, il n'y a plus de place disque ou mémoire, un conflit de fusion que tu ne sais pas résoudre, ou l'étape exige une action interdite ci-dessus. Écris alors la raison dans `propositions/AVANCEMENT.md`, commite, pousse si possible, et termine. Plafond de sécurité : 72 h de calcul au total, puis fais le bilan avec ce que tu as.

## État vérifié (25/11/2026)
- Tes 25 runs C7 sont exacts (recalculés), le protocole était commité avant ; le début de deux runs (`smax` et `dlind-elu1`, seed 101) rejoué sur une autre machine donne les mêmes valeurs ; n-gramme rejoué : 1,3402.
- À pas égaux (val entière, moyenne de 5 seeds, constaté après coup), `dlind-elu1` est devant `smax` jusqu'au pas 1 250 environ (−0,17 au pas 400), `smax` la dépasse vers le pas 1 300 et l'écart grandit (+0,022 au pas 1 800). Les 25 runs finissent à leur minimum au pas 1 800 : **1 800 pas, c'est trop court**. (Ta justification « val@1800 < val_min » compare deux mesures différentes : corrige la phrase.)
- **Défaut 1.** `data/test_oeuvres.txt` n'a pas été nettoyé comme le corpus : 13 154 apostrophes courbes (’) sont remplacées par `¤` à l'évaluation. Les chiffres « œuvres hors corpus » de C7 ne sont pas utilisables.
- **Défaut 2.** `data/fr5m` contient des passages recopiés entre train, val et test : 23 passages dans la val (jusqu'à 875 caractères) et 5 dans le test (jusqu'à 935), par exemple dans « De la Terre à la Lune » et « Madame Bovary ». Ta conclusion « pas de passage dupliqué » est à corriger. `data/fr1m` est propre (0 copie de 60 caractères).

## Étape 1 — Réparer (bloquant)
1. `data/test_oeuvres_v2.txt` : applique `normaliser` de `scripts/preparer_corpus.py` ; garde l'ancien fichier. Critère : moins de 0,1 % de caractères inconnus (attendu : 140 sur 1 234 422).
2. Réévalue les 25 runs C7 sur ce texte. Si les `checkpoint.npz` n'existent plus, relance ces 25 runs (mêmes seeds, même commit, environ 75 min) après avoir vérifié sur un run que `historique.csv` est identique. Amendement daté dans `RESULTATS_C7.md`, `RESEARCH_LOG.md` et `HYPOTHESES.md` : anciens chiffres « œuvres » marqués invalides.
3. `data/fr5m_v2` (nouveau nom, ancien conservé) : trouve pourquoi des passages sont en double, corrige le nettoyage, reconstruis. Critère : **0 fenêtre de 60 caractères** (pas de 5) de val ou test présente mot pour mot dans le train. Corrige aussi `RAPPORT_DONNEES.md`.
4. `scripts/verifier_corpus.py` : contrôle automatique d'un corpus (empreintes SHA-256 avec fins de ligne LF, vocabulaire, caractères inconnus, copies ≥ 60 caractères, textes de test nettoyés comme le corpus). Ajoute `data/** -text` dans `.gitattributes` (sous Windows, git change les fins de ligne et les empreintes ne correspondent plus).

## Étape 2 — C8 : le budget d'entraînement (question principale)
**C8 est lancé sur GitHub Actions, ne le relance pas.** Tout est déjà écrit et commité : `propositions/PROTOCOLE_C8_GITHUB.md`, `scripts/c8_github.py`, `scripts/analyse_c8_github.py`, `.github/workflows/c8.yml`.
- Pendant ce temps, fais l'étape 1, puis l'étape 3.
- Régulièrement : `git pull --rebase origin main`. Quand `results/c8g_analyse.json` est là, vérifie que les 45 runs sont complets, **recalcule les verdicts avec un second script indépendant** (`scripts/verif_c8_github.py`) et compare à `propositions/RESULTATS_C8_brut.md`.
- Écris alors `propositions/RESULTATS_C8.md`, `RESEARCH_LOG.md` et `HYPOTHESES.md` (étiquettes C8-V0 à C8-V5). Si un IC ne tranche pas, ouvre un amendement daté (5 seeds de plus : 661, 672, 683, 694, 705).

## Étape 3 — C9 : le coût (là où la méthode peut gagner)
Pour `smax`, `dlind-elu1`, `dlind-const` (la vitesse et la mémoire ne dépendent pas de la valeur des poids : les poids de C7 suffisent, la qualité vient de C8) : tokens par seconde en génération, mémoire d'état ou de cache, à T = 96, 256, 1 024 et 4 096. Utilise `experiments/profiler_inf.py` (`--phi elu1`) ; pour `dlind-const`, écris `scripts/profiler_const.py` avec le même garde-fou (écart sur les logits < 2e-3 entre forme incrémentale et forme parallèle). Donne un tableau unique « qualité (perte) / vitesse / mémoire », et le coût en pour cent de perte par rapport à `smax`.

## Étape 4 — C10 : contexte plus long
Entraîne `smax`, `dlind-elu1` et `dlind-const` avec T = 256 (batch réduit pour garder environ 1 536 caractères par pas, 5 400 pas, 3 seeds) : l'écart change-t-il quand le contexte grandit ? Protocole commité avant.

## Étape 5 — Choisir la suite (tableau de décision)
| Constat | Suite |
|---|---|
| Le meilleur réseau est encore à plus de 0,3 nat du n-gramme après C8 | capacité d'abord : d = 192, `smax` et `dlind-elu1`, 3 seeds |
| L'écart au n-gramme est passé sous 0,3 nat | données : `data/fr5m_v2`, `--pas-eval 96` |
| `smax` gagne nettement en qualité, mais `dlind-const` est bien moins chère | chercher le meilleur compromis qualité/coût (C9 et C10) et le dire clairement |
| `dlind-elu1` ou `dlind-const` devient meilleure que `smax` quelque part | confirme avec 5 seeds neufs avant de l'annoncer |
| Un IC ne tranche pas | ajoute des seeds, sinon écris « indéterminé » |

## Étape 6 — Bilan final (`propositions/BILAN_FINAL.md`)
Réponds en phrases simples : notre méthode est-elle utile, dans quel cas, à quel prix, et qu'est-ce qui reste inconnu ? Avec un tableau « entre combien de lettres le modèle hésite » (= exponentielle de la perte) pour chaque variante et le n-gramme, et une ligne « ce que je referais autrement ».

## À chaque étape
- Écris quelques lignes simples dans `propositions/AVANCEMENT.md` (fait, résultat, suite, problème), commite et pousse.
- Relis ta conclusion comme un adversaire : fuite de données, métriques différentes, hasard des seeds, autre explication ? Consigne aussi les résultats négatifs.
- Au plus un processus par cœur physique, 1 thread chacun, priorité basse ; reprise après coupure ; après trois échecs d'un run, saute-le et note-le.
- Mets à jour `RESEARCH_LOG.md`, `HYPOTHESES.md` et `RELEVE.md` (étiquettes C8-, C9-, C10-).

Commence maintenant par l'étape 1, sans attendre de réponse. Si une consigne est floue, prends l'option la plus prudente, écris pourquoi, et continue.
