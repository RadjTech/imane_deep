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
