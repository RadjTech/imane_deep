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

**Date** : 2026-11-23 · **Statut** : ✅ terminé — Décision : POURSUIVRE → C3 (delta rule) · H1b infirmée · H4 ouverte

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

### 4. Résultats

**(a) Statistique — val_min sur 3 seeds** (`results/c2/analyse.json`) :

| Configuration | s1337 | s2024 | s777 | moyenne | écart type |
|---|---|---|---|---|---|
| `smax,smax` | 2.1235 | 2.1549 | 2.1559 | **2.1448** | 0.0184 |
| `lin,lin` | 2.2349 | 2.2573 | 2.2517 | **2.2480** | 0.0117 |
| `lin,smax` | 2.1112 | 2.2265 | 2.1578 | **2.1652** | **0.0580** |

Équivalence d'implémentation (seed 1337, script unifié vs scripts du cours) : **Δ = 0.0000** sur les deux baselines — chiffre pour chiffre.

**(b) Profillage batch-1** (`results/profile_inf_*.json`, checkpoints s1337) :

V_gen (tok/s) selon la longueur de contexte T (mécanisme pur, pos_emb neutralisée) :

| T | `smax,smax` | `lin,smax` | `lin,lin` | KV smax/hybride |
|---|---|---|---|---|
| 256 | 2 404 | 2 394 | 2 494 | 384 / 192 Ko |
| 1 024 | 1 341 | 1 824 | 2 652 | 1 536 / 768 Ko |
| 4 096 | **588** | **948** | **2 693 (plat)** | 6 144 / 3 072 Ko |

Contraste naïf vs incrémental à T≈96 : 140,7 → 2 610 tok/s (`smax,smax`) = **×18,6**.
Mémoire : poids 939 Ko (identique) ; état linéaire 9,4 Ko/couche (constant) ; KV@4096 : 6,1 Mo / 3,1 Mo / 0.
Garde-fou profileur : équivalence incrémental↔parallèle ≤ 3,7e-05 sur les 3 configurations ✔.
(Bruit de timing NumPy ≈ ±20 % : ne comparer que les tendances et les rapports.)

### 5. Analyse

1. **H1b infirmée** : mean(lin,smax)=2.1652 > mean(smax,smax)=2.1448 ; victoires appariées 1/3. Le « hybride > softmax » de C1 était un hasard de seed (le s1337 favorisait l'hybride). Conséquence consignée : rétrogradation dans `HYPOTHESES.md`.
2. **H1 maintenue sur la moyenne** : rattrapage = (2.2480−2.1652)/(2.2480−2.1448) = **80 %** (critère ≥ 50 %). Mais variance de l'hybride (std 0.058) > 0.05 → **H0 déclenche « plus de seeds requis »** pour toute conclusion fine impliquant l'hybride. La sensibilité au seed du ratio est elle-même un résultat (2 couches = régime instable du placement du ratio).
3. **Résultat robuste (3/3 seeds, écart 0.103 nat ≫ std des configs pures 0.012-0.018)** : le tout-linéaire perd ~0.103 nat contre le softmax — l'état cumulatif ne récupère pas seul la capacité de routage précis.
4. **NOUVEAU résultat robuste (3/3 seeds, écart moyen 0.56 nat)** : au régime sur-appris (1 800 pas), l'hybride généralise nettement mieux (val ≈ 2.55 vs 3.11 ; train 0.82 vs 0.33). La couche à état **borne la mémorisation → régularisation implicite**. C'est un angle de recherche en soi (l'état récurrent comme contrôle du budget de mémorisation).
5. **Profilage** : constance O(1) mesurée (`lin,lin` plat à 2 693 tok/s jusqu'à T=4096, KV=0) contre la décroissance O(T) du softmax (2 404 → 588 tok/s, KV 6,1 Mo) ; l'hybride tient l'entre-deux (948 tok/s, 3,1 Mo). Le KV-cache incrémen­tal apporte ×18,6 sur la génération naïve.

### 6. Décision

- **H1b rétrogradée** (infirmée) ; **H1 maintenue** (moyenne, rattrapage 80 %) ; **H0** : seuil déclenché — conclusions fines (< 0.06 nat) impliquant l'hybride gelées sans ≥ 5 seeds.
- **POURSUIVRE → C3 (delta rule, H2)** : candidat principal pour récupérer les 0.103 nat manquants du tout-linéaire à mémoire constante.
- **Nouvelle hypothèse H4 annoncée** (voir `HYPOTHESES.md`) : l'effet régulariseur de l'état — à mesurer aussi dans C3 (val@1800 en plus de val_min dans TOUS les protocoles futurs).
- Design futur : si un écart fin doit être tranché, passer à 5 seeds (coût ~10 min/seed, assumé).
