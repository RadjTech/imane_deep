# Avancement de l'agent — journal simple

## 25/11 — Étape 1 (réparations) : FAITE ✔

**Fait.**
- Garde-fous relancés : gradients ✔, conformité φ ✔, contrôles C3b ✔.
- `data/test_oeuvres_v2.txt` : normaliser appliqué, **140 ¤ / 1 234 422 = 0.0113 %** ✔ (valeur attendue confirmée).
- 25 runs C7 réévalués sur ce texte (mêmes poids, aucun run relancé) : nouveaux chiffres « œuvres » dans l'amendement n°2 de `RESULTATS_C7.md` — ordre des variantes inchangé.
- Défaut 2 compris : répliques/scènes répétées à l'identique (paragraphes courts) passaient le filtre « paragraphe entier ≥ 60 car. ». `nettoyer_sources.py --anti-repets` écrit (fenêtres de 60 car., tous offsets, index global) ; **deux bugs corrigés au passage** : verrouillage (ré-indexation des fenêtres de queue — les œuvres suivantes étaient rejetées à 100 %) et débordement d'entier 64 bits.
- `data/fr5m_v2` reconstruit : **0 fenêtre de 60 car. dupliquée ✔** (vérifié par `scripts/verifier_corpus.py`, conforme ; fr1m re-vérifié conforme).
- `scripts/verifier_corpus.py` + `.gitattributes` (`data/** -text`) écrits.
- Amendements datés : `RAPPORT_DONNEES.md` (n°1), `RESULTATS_C7.md` (n°2, n°3), `RESEARCH_LOG.md`, `HYPOTHESES.md`.

**Résultat.** Données réparées et vérifiées ; les verdicts C7 ne changent pas.

**Suite.** Étape 3 (C9 — le coût) pendant que C8 tourne sur GitHub Actions.

## 25/11 — Étape 3 (C9, le coût) : FAITE ✔

**Fait.**
- `scripts/profiler_const.py` écrit pour `dlind-const` (état vectoriel S ← (1−β)S + βv, lecture y = S/t) avec **double garde-fou** : forme parallèle locale vs `hn.forward` installé « const » (3.8e-06) et incrémental vs parallèle (3.8e-06, seuil 2e-3 ✔).
- Profillage `smax` et `dlind-elu1` avec `experiments/profiler_inf.py --phi elu1` (checkpoints C7 s101).
- Tableau unique qualité/vitesse/mémoire : `propositions/RESULTATS_C9.md`.

**Résultat.** Voir `propositions/RESULTATS_C9.md`.

**Suite.** Surveiller C8 (`results/c8g_analyse.json`), vérifier avec `scripts/verif_c8_github.py`, puis étapes 4-6.

**Chiffres C9 (mesurés, garde-fous ✔)** :
| variante | V_gen @4096 | mémoire @4096 | coût perte vs smax |
|---|---|---|---|
| smax | 899 | KV 6 144 Ko | — |
| dlind-elu1 | 3 791 (plat) | état 19 Ko | +0.93 % |
| dlind-const | 5 513 (plat) | état 0.8 Ko | +1.49 % |
Détail : propositions/RESULTATS_C9.md. Qualité provisoire (C7) — C8 la remplacera.

## 25/11 — C8 (45 runs GitHub Actions) : REÇU ET VÉRIFIÉ ✔

**Fait.** Pull (les 45 runs + analyse officielle + rapport brut) ; intégrité 45/45 ✔ ;
vérification indépendante `verif_c8_github.py` : IDENTIQUE à l'analyse officielle ;
`RESULTATS_C8.md` + journal + hypothèses écrits et poussés.

**Résultat.** smax reste devant à tous les budgets (V1/V2/V3 infirmées) ; MAIS le meilleur
réseau (smax, 18 000 pas) est à **0.0064 nat du compteur de lettres** sur la val (C7 : 0.48)
— le tableau de décision dit : données ensuite (fr5m_v2). La version simple ne suffit plus
à 18 000 pas (+0.065 > +0.05). Coût (C9) : dlind-const reste 6× plus rapide à long contexte
avec 0.8 Ko d'état — le compromis qualité/coût reste son argument.

**Suite.** C10 (T=256, 5 400 pas, 3 seeds, protocole commité avant — local) ; puis cycle
données fr5m_v2. Plafond 72 h de calcul : ~1 h utilisée à ce jour (hors C8 GitHub).

## 26/11 — C10 analyse provisoire + C11 reçu et vérifié ✔

**Fait.**
- C10 (9 runs T=256) : analyse + vérification IDENTIQUE. V1 **indéterminé** (+0.0022, IC contient 0 — contre +0.0868 tranché à T=96 : l'avantage de smax disparaît à long contexte) ; V2 infirmée de justesse ; V3 descriptif IC disjoints. **Amendement n°1 commité avant les runs complémentaires** (744/755/766, smax+dlind-elu1 uniquement — V2 déjà tranchée, pas de shopping de verdict) ; 6 runs lancés en local (~6 h).
- C11 (GitHub Actions, 9/9, amendement distant conforme) : vérification indépendante IDENTIQUE. V1/V2 infirmées ; le compteur profite des données à budget fixe, le réseau non → le budget doit croître avec les données. `RESULTATS_C11.md` écrit.
- Push de tout (rebase sans conflit avant chaque push).

**Résultat.** Qualité : smax devant partout où c'est tranché ; dlind sans désavantage détecté à T=256 ; coût : dlind-const 6× plus rapide à long contexte, état 0.8 Ko.

**Suite.** 1) Fin de l'amendement C10 → conclusion définitive V1 à 6 seeds → push. 2) C11-bis (fr5m_v2, 18 000 pas) = à déclencher sur GitHub par l'utilisateur. 3) Bilan final (étape 6) après C10-amendement.

## 26/11 — C10 conclus + BILAN FINAL écrit ✔ (mission terminée)

**Fait.**
- Amendement C10 exécuté (6 runs, 15/15 intègres) : V1 final = indéterminé resserré (−0.0029, IC [−0.0166 ; +0.0107], 6 seeds) — disjoint de C8-T96 : l'avantage de smax disparaît à T=256. V2 inchangée (3 seeds, anti-shopping). `RESULTATS_C10.md` conclusion définitive.
- `BILAN_FINAL.md` écrit (réponses simples, tableau d'hésitation, reste inconnu, ce que je referais autrement).
- Tout poussé sur main (rebase avant chaque push).

**Suite recommandée** : C11-bis (fr5m_v2 + 18 000 pas) à déclencher sur GitHub Actions par l'utilisateur ; C6 reste réservé.
