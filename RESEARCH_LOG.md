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

**Date** : 2026-11-23 · **Statut** : ✅ terminé — H3 confirmée (Δ_A +0.018 nat, sous le bruit ; Δ_B +2.085, hasard) · Décision : POURSUIVRE → C5 (profondeur, 4 couches)

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

### 4. Résultats

**Garde-fous** : float `--check-grads` 4.27e-05 ✔ (régression) ; ternaire `--check-ternary` ✔ — quantificateur testé unitairement, FD du backward par rapport aux poids effectifs **0.00e+00**, transfert STE consigné comme par construction (le quantificateur est localement constant, le FD ne peut le valider).

**(a) Bras A — entraînement DANS la contrainte** (3 seeds, val_min, eval d'entraînement, `results/c4/analyse.json`) :

| | s1337 | s2024 | s777 | moyenne | écart type |
|---|---|---|---|---|---|
| val_min (ternaire) | 1.9095 | 1.9039 | 1.9329 | **1.9154** | 0.0125 |
| val@1800 (ternaire) | 1.9317 | 1.9039 | 1.9684 | **1.9347** | 0.026 |
| *rappel float C3* | *1.8931* | *1.8873* | *1.9124* | *1.8976* | *0.0131* |

**Δ_A = +0.0178 nat** — sous le seuil d'interprétabilité H0 (2×0.0131 = 0.0262) : à ce scale, l'entraînement ternaire est **indistinguable du float** sur val_min.

**(b) Bras B — quantification a posteriori** (checkpoints float C3 ternarisés sans ré-entraînement, eval fixe apparié) :

| ckpt | val float (eval fixe) | val ternaire | Δ |
|---|---|---|---|
| s1337 | 2.0581 | 4.1899 | **+2.1318** |
| s2024 | 2.0747 | 4.1407 | **+2.0660** |
| s777 | 2.1128 | 4.1709 | **+2.0581** |

**Δ_B = +2.0853 nat** en moyenne — val ≈ 4.14–4.19, au niveau du hasard (ln 77 = 4.34). Les 3 seeds sont d'accord (±0.04).

**(c) Mémoire et vitesse** (`results/c4/checkpoint_ternaire_s1337.npz`, garde-fou profileur 5.25e-06 ✔) : poids **939.4 Ko (float32) → 129.4 Ko packé (×7.3)** ; les poids de matmul seuls : 884.7 → 54.6 Ko (**×16.2**, 2 bits/param + échelles). V_gen plate à long contexte (2 533–2 628 tok/s @T=256..4096, KV 0) — identique au float dans NumPy : le gain « multiplications → additions entières » requiert des kernels entiers hors périmètre NumPy, consigné.

### 5. Analyse

1. **H3 confirmée massivement** : Δ_A = 0.0178 < Δ_B/2 = 1.0427. Et plus fort que le critère : Δ_A est **sous le seuil de bruit** (H0) — la contrainte {-1, 0, +1} ne coûte rien de mesurable à ce scale, contre +2.09 nat pour la quantification a posteriori. La ternarité doit être **dans** l'entraînement : confirmation expérimentale de la littérature BitNet sur notre stack.
2. **Bonus régulariseur (H4 étendue)** : le plafond d'entraînement ternaire est plus bas (train_final 1.43–1.59 vs 0.98–1.07 float) — les matmuls ternaires ne peuvent pas mémoriser aussi finement — mais la val@1800 est **meilleure** que le float (1.9347 vs 2.0281) : pente de surapprentissage **+0.019** contre +0.13 (dlind float) et +0.97 (smax). Trois mécanismes bornants se cumulent : état delta, ternarité, et leur combinaison sur-apprend à peine.
3. **Entraînement plus lent à converger** : s2024 atteint son val_min AU pas 1800 (encore en descente). À protocole figé (1 800 pas, non modifiable), le ternaire ne perd que 0.018 nat ; plus de pas pourrait combler le reste — consigné sans changement de protocole.
4. **La mission CPU-first est maintenant assemblée** : qualité > softmax (C3) + 2 bits/poids (C4) + état constant 19 Ko + KV 0 octet + poids 129 Ko. Le modèle complet tient dans **~148 Ko** de mémoire de travail et bat l'attention softmax float32 sur ce périmètre.
5. **Périmètre** : corpus 11 Ko, modèle 240 K — ceci valide la *méthode* (train-within > quantize-after, γ·Q + STE) sur notre stack, pas les claims d'échelle de BitNet (2B-4T tokens). Le gain de vitesse add-only (2.4–6× dans la littérature) est un candidat de cycle « kernels entiers ».

### 6. Décision

**C4 ✅ terminé — H3 confirmée (3/3 seeds, les deux critères secondaires inclus).** La stack CPU-first complète est validée pièce par pièce : delta rule (C3) + ternarité dans l'entraînement (C4). Décision : **POURSUIVRE → C5 (profondeur)** — la delta rule et la ternarité survivent-elles à 4 couches ? (question ouverte depuis C2 : « la carte des ratios à 4 couches » ; l'ablation de profondeur teste si les gains C3/C4 tiennent à l'échelle du modèle, avec les baselines `smax`×4 correspondantes).

---

## Cycle C5 — Profondeur : les gains survivent-ils à 4 couches ?

**Date** : 2026-11-23 · **Statut** : ✅ terminé — H5 confirmée (amplifiée : −0.333 nat), H4c confirmée (−0.697), H6 infirmée de justesse (+0.057, effet budget) · Décision : POURSUIVRE → C6 (budget : trancher H6)

### 1. Question
À profondeur doublée (4 couches, ≈ 464 K params, largeur inchangée), (1) la delta rule garde-t-elle l'avantage sur le softmax (H5) ? (2) la contrainte ternaire reste-t-elle quasi gratuite (H6) ? (3) l'effet régulariseur de l'état persiste-t-il (H4c) ?

### 2. Hypothèses annoncées avant l'expérience
- **H5** : mean(val_min)(`dlind`×4) ≤ mean(val_min)(`smax`×4) sur 3 seeds → confirmée ; écart < 2×std_max(C5) → « confirmée mais non distinguable du bruit » ; sinon → infirmée.
- **H6** : mean(val_min)(`dlind`×4 ternaire) − mean(val_min)(`dlind`×4 float) ≤ **0.05 nat** → confirmée.
- **H4c** : mean(val@1200)(`dlind`×4 float) ≤ mean(val@1200)(`smax`×4) − **0.2 nat** → confirmée.

### 3. Protocole (commité avant l'expérience)
- **Aucune modification du modèle** (le code supporte N couches) — le garde-fou gradient est **étendu à 4 couches** pour prouver la composition : FD `dlind`×4 = **3.53e-05**, `smax`×4 = **2.19e-05** ✔ (2 couches inchangées : 6.86e-05 / 4.27e-05). Garde-fou ternaire inchangé ✔.
- **3 configurations** × 3 seeds {1337, 2024, 777} : `dlind,dlind,dlind,dlind` ; `smax,smax,smax,smax` ; la même + `--ternary`. **Budget identique pour toutes : 1 200 pas** (cosinus complet, warmup 100, lr 1e-3, B=16, T=96, d=96, H=4, corpus inchangé) — **comparaisons INTERNES au cycle uniquement** (les baselines C3/C4 sont à 1 800 pas, non comparables directement, consigné).
- Params attendus : ≈ 464 K (4 × 111,844 par couche + 16,800 emb/ln ; +4 γ par couche dlind).
- Profilage V_gen batch-1 des checkpoints s1337 (`dlind`×4 vs `smax`×4) après les runs.
- Analyse pré-enregistrée : `scripts/analyse_c5.py` ; exécution : `scripts/run_c5.sh`.

### 4. Résultats

**Garde-fous** : FD étendus aux 4 couches avant les runs — `dlind`×4 **3.53e-05**, `smax`×4 **2.19e-05** ✔ (2 couches inchangées : 6.86e-05 / 4.27e-05). Ternaire ✔.

**(a) Statistique — 3 configs × 3 seeds, 1 200 pas, 464 K params** (`results/c5/analyse.json`) :

| Configuration | s1337 | s2024 | s777 | moyenne | écart type | val@1200 |
|---|---|---|---|---|---|---|
| `dlind`×4 | 1.8306 | 1.8414 | 1.8601 | **1.8441** | 0.0122 | 1.8826 |
| `dlind`×4 ternaire | 1.9133 | 1.8937 | 1.8960 | 1.9010 | **0.0088** | 1.9083 |
| `smax`×4 | 2.1724 | 2.1926 | 2.1647 | 2.1766 | 0.0118 | 2.5796 |

Seuil de bruit interne H0 : 2×std_max = **0.0244** nat. Pas de val_min du `dlind`×4 : 900–1100 ; du ternaire : 1100–**1200** (s2024 au tout dernier pas) ; du `smax`×4 : **500–600** (effondrement précoce).

**(b) Profillage batch-1** (checkpoints s1337, garde-fous 6.2e-06 / 1.75e-05 ✔) :

| Contexte T | `dlind`×4 | `smax`×4 |
|---|---|---|
| 256 | 1 363 | 1 187 |
| 1 024 | 1 329 | 776 |
| 4 096 | **1 313 (plat)** | **244** |

KV `smax`×4 : **12,3 Mo à T=4096** (3 072 octets/token × 4 couches) ; `dlind`×4 : 0 octet, état constant 38 Ko. Ratio de vitesse à long contexte : **×5.4** (×4.6 à 2 couches).

### 5. Analyse

1. **H5 confirmée — supériorité AMPLIFIÉE par la profondeur** : écart dlind−smax de **−0.333** nat (bruit 0.024), contre −0.247 à 2 couches. La profondeur aide la delta rule (val_min 1.8976 → 1.8441, −0.054 vs 2 couches) et **nuit** au softmax : le `smax`×4 atteint son val_min dès le pas 500–600 puis s'écroule (train 0.86–0.99 = mémorisation, val@1200 2.58). Plus de couches softmax = plus de capacité de mémorisation du corpus de 11 Ko, pas plus de généralisation.
2. **H4c confirmée — régulariseur amplifié** : écart val@1200 de **−0.697** (−0.56 à 2 couches). Pente de surapprentissage (val@1200 − val_min) : **+0.007** (ternaire), +0.039 (`dlind`×4), **+0.403** (`smax`×4). Trois mécanismes bornants (état delta, ternarité, profondeur × état) se cumulent sans se dégrader.
3. **H6 infirmée de justesse** : Δ_tern = **+0.057** > 0.05 (contre +0.018 à 2 couches). **Nuance consignée** : les 3 runs ternaires atteignent leur val_min tard et sont **encore en descente au dernier pas** (s2024 : min au pas 1200 exactement) — l'effet budget (1 200 pas) est probable, mais le critère pré-enregistré reste le contrat : à budget égal, le coût du ternaire à 4 couches n'est pas ≤ 0.05. Question ouverte → C6.
4. **La stack CPU-first tient la profondeur** : le modèle **tout-ternaire 4 couches** (1.9010, poids ≈ 175 Ko packé (×10.4 vs float 1 813 Ko), état 38 Ko, KV 0) bat le softmax float32 à 2 couches (2.1448) ET le softmax float32 à 4 couches (2.1766), à budget inférieur (1 200 vs 1 800 pas).
5. **Coûts** : entraînement `dlind`×4 ≈ 1 950–2 090 tok/s (moitié du 2 couches — séquentiel BPTT ×4 couches, coût d'implémentation NumPy consigné) ; en génération longue, `dlind`×4 est plat (~1 300 tok/s) là où `smax`×4 tombe à 244.

### 6. Décision

**C5 ✅ terminé — H5 confirmée (amplifiée), H4c confirmée (amplifié), H6 infirmée de justesse (+0.057, effet budget probable mais consigné).** La delta rule n'est pas seulement robuste à la profondeur : son avantage **grandit** avec elle. Décision : **POURSUIVRE → C6 (budget)** — trancher la question ouverte par H6 : le coût du ternaire à 4 couches est-il un artefact de budget ? (`dlind`×4 float vs ternaire à **2 400 pas**, 3 seeds, critère H6b pré-enregistré à l'ouverture du cycle).

---

## Cycle C3b — Re-mesurer les fondations : φ, normalisation des clés, oubli

**Date** : 2026-11-24 · **Statut** : ✅ terminé (C3b-H5 indéterminée, C3b-H9 infirmée de peu) · **Décision proposée** : PIVOTER vers un corpus plus grand avant de conclure sur le mécanisme

> Hypothèses et analyse pré-enregistrées dans `propositions/PROTOCOLE_C3b.md` et `scripts/analyse_c3b.py` (commits `7c61e39` et `109319f`, avant le premier run),
> et non dans `HYPOTHESES.md` (écart au point 2 du protocole, consigné).
> **Étiquettes locales** `C3b-H5` … `C3b-H9` : H5 et H6 existent déjà pour C5 avec un autre sens.

### 1. Question
Les conclusions de C1–C3 survivent-elles à une φ conforme (ELU+1) ? D'où vient le gain de `dlind` sur `lin` : normalisation des clés, oubli, ou adressage par contenu ?

### 2. Hypothèses annoncées avant l'expérience
- **C3b-H5** : IC 95 % de (`dlind-const` − `dlind-hist`) ⊂ [−0,05 ; +0,05].
- **C3b-H6** : IC 95 % de (`dlind-elu1` − `dlind-hist`) entièrement > 0.
- **C3b-H7** : IC 95 % de (`dlind-elu1` − `smax`) entièrement < 0.
- **C3b-H8** : IC 95 % de (`lin-hist-nk` − `lin-hist`) ⊂ [−0,05 ; +0,05].
- **C3b-H9** : (`lin-elu1` − `hybride-elu1`) ≥ 0,5 × (`lin-elu1` − `smax`), avec IC 95 % de l'écart `lin-elu1` − `smax` > 0.

### 3. Protocole (commité avant l'expérience)
9 variantes × 5 seeds neufs (11, 22, 33, 44, 55) = 45 runs, 1 800 pas, réglages C2/C3, métrique = minimum de la perte sur la **val entière** (`val_entiere_min`), comparaisons appariées par seed.
`python scripts/lancer_parallele.py --cycle c3b --variantes --configs smax hybride-hist hybride-elu1 lin-hist lin-elu1 lin-hist-nk dlind-hist dlind-elu1 dlind-const --seeds 11 22 33 44 55` puis `python scripts/analyse_c3b.py`.

### 4. Résultats (val entière min, moyenne de 5 seeds)

| Variante | moyenne | écart type (ddof = 1) | val @1800 |
|---|---|---|---|
| `smax` | 2.1613 | 0.0219 | 3.1032 |
| `hybride-hist` | 2.1533 | 0.0297 | 2.5365 |
| `hybride-elu1` | 2.2004 | 0.0204 | 3.0425 |
| `lin-hist` | 2.2543 | 0.0073 | 2.7569 |
| `lin-elu1` | 2.2347 | 0.0118 | 3.1427 |
| `lin-hist-nk` | 2.2622 | 0.0096 | 2.5180 |
| `dlind-hist` | 1.9021 | 0.0199 | 2.0238 |
| `dlind-elu1` | 1.9967 | 0.0094 | 2.4471 |
| `dlind-const` | **1.8591** | 0.0219 | **1.9168** |

Référence non neuronale (n-gramme Kneser-Ney, même val) : **1.6827**.

| # | Différence | IC 95 % | Verdict |
|---|---|---|---|
| C3b-H5 | −0.0430 | [−0.0617 ; −0.0243] | indéterminée (équivalence à ±0,05 non atteinte) |
| C3b-H6 | +0.0946 | [+0.0585 ; +0.1307] | confirmée |
| C3b-H7 | −0.1646 | [−0.1976 ; −0.1316] | confirmée |
| C3b-H8 | +0.0079 | [−0.0066 ; +0.0224] | confirmée |
| C3b-H9 | 47 % (seuil 50 %) | contraste −0.0024, IC [−0.0343 ; +0.0295] | infirmée de peu, dans le bruit |

Vérifications : 45 runs complets, même commit `109319f` (code non modifié), calculs refaits par un second script, rejeu exact (octets et poids) de `dlind-const` s33 et `dlind-hist` s33. Détail : `propositions/RESULTATS_C3b.md`.

### 5. Analyse
1. `dlind` reste devant `smax` avec la φ correcte (5/5 seeds) : la conclusion de C3 tient sur ce point, avec un écart plus petit (−0,165 au lieu de −0,26).
2. La φ historique aidait `dlind` (+0,095 sans elle, et plus de sur-apprentissage : val@1800 2,02 → 2,45) : H4/H4b sont à ré-attribuer en partie à φ.
3. `dlind-const` (sans requête ni clé) fait au moins aussi bien que `dlind` (−0,043, 5/5 seeds). Le gain de `dlind` n'exige pas d'adressage par contenu sur ces données. Lecture faite après coup ; le verdict pré-enregistré est « indéterminée ».
4. H9 : le ratio de rattrapage (47 %, seuil 50 %) dépend de la φ ; il n'est pas au-dessus de 50 % de façon fiable (IC du contraste contient 0, seed 55 négatif).
5. Aucune variante n'égale le n-gramme (1,683) : à 11 Ko, ces écarts mesurent probablement surtout la régularisation.

### 6. Décision
**Proposée : PIVOTER vers un corpus plus grand** (train / val / test séparés) avant toute conclusion sur le mécanisme. La suite est décrite dans `propositions/PROMPT_RELEVE.md`. Le passage à 8 seeds prévu pour H5 et H9 n'a pas été lancé (15 runs, environ 1 h sur la machine locale) : à décider.

---

## Cycle C7 — Validation sur corpus réel (fr1m, 998 K caractères, wikisource)

**Date** : 2026-11-24 · **Statut** : ✅ terminé — C7-V1 INFIRMÉE (smax bat dlind à 1 Mo), C7-V2 confirmée (la version simple suffit), C7-V3 : la φ corrigée est meilleure · **Étiquettes locales C7-V…** (H5/H6 déjà pris par C5 et C3b)

> Protocole + scripts pré-enregistrés commités avant tout run (`43f9144`) ; données :
> `propositions/RAPPORT_DONNEES.md` + `data/SOURCES.md` (fr.wikisource.org, gutenberg.org
> 503 depuis ce bac à sable) ; test_oeuvres = Huysmans + About (hors corpus).
> Vérification indépendante : IDENTIQUE (4.9e-05) après correction d'un bug de quantile
> dans le second script (amendement n°1, avant toute lecture des verdicts).

### 1. Question
Les conclusions de C3b (corpus 11 Ko) tiennent-elles sur 1 Mo de prose française réelle ?
### 2. Critères pré-enregistrés (avant les runs)
- **C7-V1** : IC(dlind-elu1 − smax) < 0 sur le test → confirmée.
- **C7-V2** : borne haute de IC(dlind-const − dlind-elu1) ≤ +0.05 → confirmée.
- **C7-V3** : signe de IC(dlind-elu1 − dlind-hist).
- **C7-V4** : descriptif — meilleur réseau vs n-gramme (même val).

### 3. Protocole
5 variantes {smax, lin-hist, dlind-hist, dlind-elu1, dlind-const} × 5 seeds {101, 202, 303, 404, 505},
1 800 pas, réglages C3b, corpus fr1m ; métrique principale = TEST (une évaluation, poids
finaux) ; secondaires : val, test_oeuvres, vitesse ; n-gramme en référence.

### 4. Résultats (test, moyenne ± écart type)

| smax | dlind-elu1 | dlind-hist | dlind-const | lin-hist | n-gramme (val) |
|---|---|---|---|---|---|
| **1.8223 ± 0.0074** | 1.8392 ± 0.0075 | 1.8466 ± 0.0069 | 1.8495 ± 0.0061 | 2.1611 ± 0.0057 | **1.3402** |

| # | différence | IC 95 % | verdict |
|---|---|---|---|
| C7-V1 | dlind-elu1 − smax = **+0.0169** | [+0.0122 ; +0.0216] | **infirmée** (smax mieux, 5/5) |
| C7-V2 | dlind-const − dlind-elu1 = +0.0102 | [+0.0023 ; +0.0182] | **confirmée** |
| C7-V3 | dlind-elu1 − dlind-hist = **−0.0074** | [−0.0138 ; −0.0009] | **φ corrigée meilleure** |
| C7-V4 | 1.8166 vs 1.3402 (val) | — | le n-gramme bat tous les réseaux |

### 5. Analyse
1. Le verdict C3b (dlind > smax à 11 Ko) ne se généralise pas : à 1 Mo, smax gagne de
   0.017 sans chevauchement des seeds — les mesures petit-corpus mesuraient surtout la
   régularisation (cf. n-gramme 1.683 à 11 Ko).
2. Ce qui tient : lin reste très loin (écart lin→dlind = 0.32, ×15 l'écart dlind↔smax) et
   dlind-const suit dlind-elu1 à +0.01 → l'essentiel du gain est l'oubli borné, pas
   l'adressage par contenu.
3. La φ corrigée redevient la meilleure avec de vraies données (à 11 Ko, la φ historique
   faisait régulariseur).
4. Tous les modèles sont en sous-apprentissage (val@1800 < val_min, 5/5 variantes) : le
   budget 1 800 pas borne la comparaison ; le n-gramme (1.34) reste le champion absolu.
   Détail complet : `propositions/RESULTATS_C7.md`.

### 6. Décision
C7 ✅ terminé — résultat négatif consigné comme tel : **à 1 Mo, la delta rule ne bat pas
l'attention softmax** (écart petit mais significatif) ; la version simple suffit ; la φ
corrigée est retenue pour les cycles futurs. Suites proposées (non lancées, à arbitrer) :
fr5m avec `--pas-eval`, budget d'entraînement étendu (sous-apprentissage général),
capacité doublée pour attaquer le plancher du n-gramme (1.34). C6 (budget) reste réservé.

**Amendements n°2 et n°3 (25/11/2026, voir `propositions/RESULTATS_C7.md`)** : (n°2) les
chiffres « œuvres hors corpus » étaient invalides (texte de test non nettoyé : 13 294 ¤) ;
réparés sur `test_oeuvres_v2` avec les mêmes checkpoints — l'ordre des variantes est
inchangé, les verdicts tiennent. (n°3) correction d'une phrase : les 25/25 runs finissent
**à leur minimum** au pas 1 800 sur la val entière (ma phrase « val@1800 < val_min »
comparait deux mesures différentes) → 1 800 pas trop court → C8 (budget) lancé sur GitHub
Actions. Deux défauts de données réparés : `test_oeuvres_v2` (140 ¤) et `fr5m_v2`
(0 copie de 60 caractères, `scripts/verifier_corpus.py` conforme) — voir
`propositions/RAPPORT_DONNEES.md` amendement n°1.

---

## Cycle C8 — Le budget d'entraînement (GitHub Actions, 45/45 runs) (2026-11-25)

**Statut** : ✅ terminé — C8-V0 confirmée (réplication : smax devant à 1 800 pas) ; C8-V1/V2/V3 INFIRMÉES (smax reste devant à 5 400 et 18 000 ; la version simple ne suffit plus à 18 000) ; C8-V4 : l'avantage de smax grandit avec le budget ; C8-V5 (descriptif) : smax 1.3466 sur la val, à **0.0064 nat du n-gramme** (1.3402) — écart passé de 0.48 (C3b) à 0.006.

> Protocole commité avant les runs (46daced) ; vérification indépendante IDENTIQUE
> (`scripts/verif_c8_github.py` → `results/c8g_verification.json`) ; aucun IC indéterminé,
> pas d'amendement seeds. Détail : `propositions/RESULTATS_C8.md`.

### Résultats (test_best, moyenne ± écart type, 5 seeds)

| variante | 1 800 | 5 400 | 18 000 |
|---|---|---|---|
| smax | 1.8204 ± 0.0097 | 1.4935 ± 0.0072 | **1.3577 ± 0.0041** |
| dlind-elu1 | 1.8352 ± 0.0020 | 1.5803 ± 0.0037 | 1.3942 ± 0.0026 |
| dlind-const | 1.8492 ± 0.0021 | 1.6236 ± 0.0039 | 1.4588 ± 0.0019 |

Différences appariées (dlind-elu1 − smax) : +0.0148 → +0.0868 → +0.0364 (forme non monotone, culmine à 5 400) ; dlind-const − dlind-elu1 @18 000 = +0.0646 (> seuil +0.05).

### Analyse
1. Plus de budget ne fait pas gagner la delta rule (3 IC sans chevauchement, 5/5 seeds) — résultat négatif consigné.
2. La qualité absolue explose : smax 1.82 → 1.3577 ; tous les runs finissent à leur minimum (pas_best = dernier pas) → encore du gain disponible.
3. Le n-gramme n'est plus une borne : 0.006 nat sur la val — le tableau de décision indique **données ensuite** (fr5m_v2, --pas-eval 96).

### Décision
C8 ✅ terminé. Suite : C10 (T=256, protocole commité avant, local) + préparation du cycle données fr5m_v2. C6 reste réservé.

---

## Cycle C11 — Plus de données (fr5m_v2, GitHub Actions, 9/9 runs) (2026-11-26)

**Statut** : ✅ terminé — C11-V1 INFIRMÉE (smax devant : +0.0909 [+0.0818 ; +0.1000], compatible avec C8-fr1m +0.0868 → les données ne changent pas le duel à budget égal) ; C11-V2 infirmée (version simple +0.0473, borne +0.0594 > +0.05) ; C11-V3 (descriptif) : écart réseau↔compteur **0.252** à b5400 — le compteur profite des données (1.3402→1.222), le réseau sous-budget n'en profite qu'à peine (1.8 époques vs 9.7) ; **le budget doit croître avec les données**.

> Amendement n°1 du prédécesseur : exécution GitHub Actions (9 runs parallèles), protocole
> inchangé, n-gramme mesuré en job séparé (justifié). Vérification indépendante IDENTIQUE
> (2.9e-05). Détail : `propositions/RESULTATS_C11.md`.

### Résultats (test_best, 3 seeds)

| smax | dlind-elu1 | dlind-const | n-gramme fr5m_v2 (val) |
|---|---|---|---|
| **1.4849 ± 0.0040** | 1.5758 ± 0.0030 | 1.6231 ± 0.0023 | 1.222 |

### Décision
C11 ✅ terminé. Prochain levier : fr5m_v2 + 18 000 pas (budget proportionné aux données,
~3× C8-18k, sur GitHub Actions — à déclencher par l'utilisateur). La capacité (d=192)
n'est pas déclenchée par le tableau de décision. En local : amendement C10 en cours
(seeds 744/755/766, smax + dlind-elu1, V1 indéterminé à 3 seeds).

**Amendement C10 exécuté (26/11)** : seeds 744/755/766 (smax + dlind-elu1, T=256) — 15/15
runs intègres, vérification indépendante IDENTIQUE (4.9e-05). **C10-V1 final :
indéterminé resserré** (−0.0029, IC [−0.0166 ; +0.0107], 6 seeds — disjoint de C8-T96
[+0.0787 ; +0.0950]) : à T=256, smax et dlind-elu1 sont indiscernables à budget égal,
alors que smax gagnait de +0.087 à T=96. V2 inchangée (infirmée, 3 seeds, anti-shopping).
→ `RESULTATS_C10.md` mis à jour (conclusion définitive).
