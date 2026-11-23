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

---

## Cycle C3 — Delta rule : l'état qui remplace au lieu d'additionner

**Date** : 2026-11-23 · **Statut** : ✅ terminé — H2 + H4b confirmées · Décision : POURSUIVRE → C4 (ternaire, H3)

### 1. Question
La mise à jour delta `S += β·k⊗(v − S·k)` — qui écrase la trace similaire au lieu d'additionner — récupère-t-elle la qualité manquante du tout-linéaire (résidu de 0.103 nat mesuré en C2) tout en gardant l'état constant et l'effet régulariseur (H4) ?

### 2. Hypothèses annoncées avant l'expérience
- **H2** (critère calibré C2) : mean(val_min)(`dlind,dlind`) ≤ **2.1964** (rattrapage ≥ 50 %).
- **H4b** : mean(val@1800)(`dlind,dlind`) ≤ **2.914** (la régularisation survit à la delta rule).

### 3. Protocole (commité avant l'expérience)
- Nouveau type de couche `dlind` dans `experiments/hybrid_numpy.py` : **seule la mise à jour de S change** par rapport à `lin` (Z, φ, normalisation identiques). β = sigmoid(γ) appris par tête (+8 paramètres, +0.003 %).
- Forme d'entraînement : séquentielle (boucle sur T) + **BPTT écrit à la main** — garde-fou `--check-grads` étendu aux types `dlind,dlind` (obligatoire avant tout run). La forme parallèle chunked (Yang et al. 2024) est hors périmètre C3, consigné comme coût d'implémentation.
- 3 seeds {1337, 2024, 777} × `dlind,dlind`, 1 800 pas, tout le reste identique à C2. Baselines de référence : moyennes C2 (`smax,smax` 2.1448/3.1142, `lin,lin` 2.2480/2.7860).
- Profilage batch-1 du checkpoint s1337 (V_gen à T ∈ {256, 1024, 4096}, mémoire d'état) — comparé au `lin,lin` plat (2 693 tok/s).
- Analyse pré-enregistrée : `scripts/analyse_c3.py`.
- Exécution : `scripts/run_c3.sh`.

### 4. Résultats

**Garde-fou** : gradients `dlind,dlind` validés avant tout run — erreur FD **4.27e-05** (lin+smax : 6.86e-05). Trois bugs trouvés et corrigés par le garde-fou avant l'expérience : clés non normalisées (divergence exponentielle, gradients ~1e10), facteur β appliqué en double dans les adjoints, ordre des adjoints (le pass-through `q⊗dnum` polluait les autres termes). Le profileur d'inférence a eu son propre bug (mise à jour delta avec le k brut au lieu du k̂ normalisé, équivalence 0.589 ✘ → corrigé, 7.15e-07 ✔).

**(a) Statistique — 3 seeds** (`results/c3/analyse.json`) :

| Configuration | s1337 | s2024 | s777 | moyenne | écart type |
|---|---|---|---|---|---|
| `dlind,dlind` | **1.8931** | **1.8873** | **1.9124** | **1.8976** | **0.0131** |
| `lin,lin` (C2) | 2.2349 | 2.2573 | 2.2517 | 2.2480 | 0.0117 |
| `smax,smax` (C2) | 2.1235 | 2.1549 | 2.1559 | 2.1448 | 0.0184 |

val@1800 (régime sur-appris) : dlind {1.9932, 2.0075, 2.0838} moy. **2.0281** vs lin 2.7860 et smax 3.1142.

**(b) Profillage batch-1** (checkpoint s1337 entraîné, garde-fou 9.18e-06 ✔, `results/profile_inf_dlind-dlind.json`) :

| Contexte T | `dlind,dlind` | `lin,lin` | `smax,smax` (C2) |
|---|---|---|---|
| 256 | 2 404 | 2 494 | 2 404 |
| 1 024 | 1 789 | 2 652 | 1 341 |
| 4 096 | **2 583** | **2 693** | 588 |

KV-cache **0 octet** (état constant 19 Ko) — profil plat comme le linéaire, sans son plafond de qualité.

### 5. Analyse

1. **H2 confirmée — et au-delà du critère** : mean(val_min) = 1.8976 ≤ 2.1964. Le delta rule ne rattrape pas 50 % de l'écart lin↔smax, il le **dépasse de 340 %** : 1.8976 est **0.247 nat sous** le softmax (2.1448). À paramètres égaux (240 488), la couche à état delta bat l'attention softmax en qualité sur ce périmètre.
2. **H4b confirmée — la régularisation non seulement survit, elle s'amplifie** : val@1800 = 2.0281 ≤ 2.914. L'écart val@1800 − val_min mesure la pente de surapprentissage : **+0.13** (dlind) contre +0.54 (lin) et +0.97 (smax). L'état qui remplace borne la mémorisation bien plus fort que l'état qui additionne — cohérent avec H4 et avec la mécanique (écraser la trace similaire ≠ accumuler).
3. **Variance la plus faible des trois configurations** (std 0.0131) : le choix de la mise à jour est robuste au seed, contrairement à l'hybride `lin,smax` (std 0.0580).
4. **β appris ≈ 0.54–0.60 sur les 8 têtes** (γ ∈ [0.16, 0.41] au checkpoint s1337) : le modèle remplace ~57 % de la trace à chaque écriture — choix cohérent entre têtes et couches, ni 0 (pur cumul = lin) ni 1 (pur remplacement).
5. **Coûts asymétriques** : entraînement plus lent que les baselines (≈ 3 950–4 330 tok/s vs ≈ 5 500–6 200 pour smax) à cause de la forme séquentielle + BPTT — coût d'implémentation NumPy, pas conceptuel (forme chunked de Yang et al. 2024 hors périmètre, consignée). En **génération**, l'avantage s'inverse : plat à T=4096 (2 583 tok/s, 0 octet de KV) là où smax s'effondre (588).
6. **Périmètre** : corpus 11 Ko, modèle 240 K, 2 couches — la supériorité qualitative sur smax est une mesure à ce scale, pas une loi générale ; le critère H2 reste le contrat.

### 6. Décision

**C3 ✅ terminé — H2 et H4b confirmées (3/3 seeds).** La delta rule normalisée devient la **couche d'état de référence** du programme : qualité > softmax, mémoire constante, génération plate à long contexte, variance de seed minimale. Décision : **POURSUIVRE → C4 (H3 ternaire)** — l'entraînement *dans* la contrainte {-1, 0, +1} sur la base `dlind,dlind`, pour convertir ce gain de qualité en gain mémoire/bande passante réels. La forme parallèle chunked (Yang et al. 2024) reste un candidat de cycle « ingénierie » pour l'entraînement.

---

## Cycle C4 — Ternarisation façon BitNet : la qualité en 2 bits ?

**Date** : 2026-11-23 · **Statut** : 🔎 en cours

### 1. Question
Le modèle `dlind` (C3) peut-il être entraîné **avec** ses poids de matmul contraints à {−1, 0, +1} (échelle γ par tenseur, straight-through estimator) en gardant une val proche du float — et la quantification a posteriori du même checkpoint est-elle bien pire, comme l'affirme la littérature BitNet (la ternarité doit être *dans* l'entraînement, pas après) ?

### 2. Hypothèses annoncées avant l'expérience
- **H3** (annoncée à l'ouverture du programme, critère fixé) : Δ(entraînement-dans-la-contrainte) < Δ(quantification-a-posteriori) / 2 → confirmée ; sinon → infirmée. Attentes secondaires : Δ_A < 0.15 nat ; Δ_B > 0.30 nat.
- Définitions fixées avant les runs : **Δ_A** = mean(val_min)(`dlind,dlind` ternaire, 3 seeds) − **1.8976** (baseline C3 float, même eval d'entraînement) ; **Δ_B** = moyenne par seed de (val_ternaire − val_float) sur un **eval fixe déterministe** (fenêtres non chevauchantes sur toute la val), checkpoints C3 ternarisés post-hoc.

### 3. Protocole (commité avant l'expérience)
- **Méthode BitNet b1.58 (absmean)** : pour chaque tenseur de poids de matmul {`qkv`, `proj`, `fc1`, `fc2`} × 2 blocs : γ = mean(|W|), Q = round(clip(W/γ, −1, 1)) ∈ {−1, 0, +1}, W_eff = γ·Q. Backward : **STE** (dW_eff appliqué au W maître tel quel). Embeddings, biais, LN et γ restent float (E/S haute précision, littérature). Implémentation : **P_effectif recalculé à chaque forward** — le code float validé est réutilisé tel quel, aucune nouvelle voie de gradient.
- **Garde-fou C4** (`--check-ternary`, obligatoire avant tout run) : (a) tests unitaires du quantificateur (valeurs ∈ {−1,0,+1}, γ>0, garde tenseur nul) ; (b) FD du backward **par rapport à P_eff** (continus → FD valide) < 1e-4 relatif, avec plancher absolu 1e-8 pour le bruit d'annulation du FD ; (c) le transfert STE maître←dW_eff est par construction et non FD-ifiable (le quantificateur est localement constant) — consigné honnêtement. Régression float `--check-grads` : 4.27e-05 ✔.
- **Bras A** : 3 seeds {1337, 2024, 777} × `dlind,dlind --ternary`, 1 800 pas, tout le reste identique à C3 (mêmes données, optimiseur, planning lr, batch).
- **Bras B** : les 3 checkpoints float C3 ternarisés **sans ré-entraînement** (`--quantize-after`), évalués avec l'eval fixe ; Δ_B apparié par seed (même ckpt, même eval).
- **Mémoire** mesurée depuis le checkpoint : 2 bits/paramètre ternaire packé + 4 octets d'échelle par tenseur + reste en float32. **Profilage** V_gen du checkpoint ternaire exporté (s1337, γ·Q + échelles).
- Analyse pré-enregistrée : `scripts/analyse_c4.py` ; exécution : `scripts/run_c4.sh`.

### 4. Résultats — à remplir à la fin des runs

### 5. Analyse — à remplir

### 6. Décision — à remplir
