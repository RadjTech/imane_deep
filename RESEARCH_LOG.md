# Journal de recherche — imane_deep

> Chaque entrée suit le protocole (`RESEARCH_PROTOCOL.md`, les 9 étapes). Les logs bruts et CSV sont dans `results/`.

---

## Cycle C1 — Combien de couches softmax suffisent ?

**Date** : 2026-11-23 · **Statut** : ✅ terminé · **Décision** : POURSUIVRE → C2 (profilage d'inférence) + C3 (delta rule)

### 1. Question
Dans un transformer de 2 couches (240 480 paramètres), quelle part de la qualité du softmax (val 2.1214) l'hybride `lin,smax` récupère-t-il par rapport au tout-linéaire (val 2.2349) — et à quel coût mémoire ?

### 2. Hypothèse (H1)
`lin,smax` rattrape ≥ 50 % de l'écart : val_min ≤ **2.175**.

### 3. Protocole
Identique pour les trois configurations — seul le type d'attention par couche varie :
`python3 experiments/hybrid_numpy.py --train --types <T> --max-steps 1800 --seed 1337`
- 240 480 paramètres (init GPT-2 N(0, 0.02), projections résiduelles réduites 1/√(2L))
- corpus 11 172 caractères, vocab 77, batch 16 × T 96
- AdamW (wd 0.05, warmup 100, cosinus → 10 %), clip 1.0
- garde-fou : backward vérifié par différences finies — écart max **6.86e-05** (types mixtes, les deux chemins couverts)

### 4. Résultats

| Configuration | val min | au pas | val @1800 | train @1800 | M@T=96 (inférence) | M@T=4096 |
|---|---|---|---|---|---|---|
| `smax,smax` (baseline) | 2.1235 | 700 | 3.1385 | 0.3046 | ≈ 147 Ko | ≈ 6.3 Mo (croît) |
| **`lin,smax` (C1)** | **2.1112** | 700 | **2.4596** | 0.7635 | ≈ 147 Ko | ≈ 3.2 Mo (croît ÷2) |
| `lin,lin` (baseline) | 2.2349 | 500 | 2.8330 | 0.9393 | ≈ 147 Ko | **≈ 147 Ko constant** |

Sources : `results/c1_synthese.json`, CSV bruts dans `results/`. Garde-fou : écart gradients max 6.86e-05. Durée C1 : ~9 min sur 2 CPU.

### 5. Analyse
- **H1 confirmée — au-delà du critère.** Critère annoncé : val_min ≤ 2.175. Obtenu : **2.1112**. L'hybride récouvre **plus que 100 % de l'écart** (0.1237 nat récupérés sur un écart total de 0.1114) : il dépasse même le tout-softmax de 0.0123 nat.
- **Mécanisme proposé** : la couche linéaire en bas *compresse* le contexte dans un état borné — ce qui limite la mémorisation (train@1800 : 0.76 vs 0.30 pour le softmax pur) et régularise les représentations ; la couche softmax en haut fait le routage précis sur ces représentations déjà agrégées. Résultat : val finale @1800 de **2.4596 contre 3.1385** au softmax pur — l'hybride généralise nettement mieux au même budget d'entraînement.
- **Coût mémoire** : à T=96 toutes les configurations tiennent dans ≈ 147 Ko ; la différence se joue dans la *croissance* : ÷2 à T=4096 pour l'hybride (une seule couche à KV-cache), constant pour le tout-linéaire. Le vrai verdict sera V_gen (C2).
- **Honnêteté (H0)** : tout ceci repose sur **1 seed**. L'écart hybride→softmax (0.0123 nat) est *sous* le seuil de bruit H0 estimé (0.03) → statut « indication » jusqu'aux 3 seeds de C2. En revanche le rattrapage du tout-linéaire (0.12 nat) est bien au-dessus du bruit : **H1 robuste à cet égard**.

### 6. Prochaines étapes / Décision
**POURSUIVRE.** Deux suites parallèles :
1. **C2 — consolidation + profilage** : 3 seeds × 3 configurations (H0 : quantifier le bruit réel), puis profiler l'inférence batch-1 réelle (V_gen, mémoire) à T ∈ {96, 1024, 4096}. Si l'écart hybride>softmax survit à 3 seeds → c'est un résultat publiable à cette échelle contrôlée.
2. **C3 — delta rule** (H2) : faire *remplacer* l'état au lieu d'additionner, pour pousser le tout-linéaire vers la val de l'hybride avec mémoire constante.

---
*Fin du cycle C1 — traçabilité : commits `C1:`, CI garde-fous active.*

---

## Cycle C2 — 3 seeds × 3 configurations + profilage d'inférence batch-1

**Date** : 2026-11-23 · **Statut** : 🔎 en cours

### 1. Question
(1) La supériorité de l'hybride `lin,smax` sur `smax,smax` observée en C1 survit-elle au bruit de seed (H1b) ? (2) Quel est le profil réel d'inférence batch-1 des trois mécanismes : V_gen mesuré, mémoire d'état, scaling en longueur de contexte ?

### 2. Hypothèses annoncées avant l'expérience
- **H0** (bruit) : std inter-seeds de val_min attendu dans [0.02, 0.04] nat — critères de décision dans `HYPOTHESES.md`.
- **H1b** (consolidation) : mean(val_min)(lin,smax) ≤ mean(val_min)(smax,smax) sur 3 seeds → confirmée ; sinon le résultat C1 est rétrogradé.

### 3. Protocole (commité avant l'expérience)
- **9 points statistiques** : {smax,smax ; lin,lin ; lin,smax} × seeds {1337, 2024, 777} via `experiments/hybrid_numpy.py` (1800 pas, identique C1). Le point (lin,smax, 1337) est réutilisé de C1 (même script, même seed).
- **Test d'équivalence d'implémentation** : le run (smax,smax, 1337) du script unifié doit reproduire le run C1 du script du cours à < 5e-3 nat près (sinon → tous les points re-run avec un seul script).
- **Profilage** (`experiments/profiler_inf.py`, garde-fou : équivalence incrémental↔parallèle < 1e-3 sur les logits) :
  - V_gen mesuré à T0 ∈ {16, 48, 96} (dans la distribution d'entraînement) ;
  - scaling mécanisme à T ∈ {256, 1024, 4096} avec pos_emb neutralisé (hors distribution *qualité* — mesure du mécanisme seul, documenté comme correction de périmètre : modèles entraînés à T=96) ;
  - mémoire : poids, état linéaire constant, KV-cache par longueur ;
  - contraste : génération naïve (re-calcul de la fenêtre) vs KV-cache incrémental.
- **Analyse pré-enregistrée** : `scripts/analyse_c2.py` (moyennes ± std, victoires appariées par seed, verdicts selon les critères H0/H1b).
- Exécution : `scripts/run_c2.sh` (8 runs restants, séquentiels, env identique à C1).

### 4. Résultats — à remplir à la fin des runs

### 5. Analyse — à remplir

### 6. Décision — à remplir
