# Hypothèses falsifiables

> Format : chaque hypothèse porte un critère de décision numérique **annoncé avant** l'expérience. Statuts : `à tester` → `confirmée` / `infirmée` / `indication` (1 seed) → `établie` (≥ 3 seeds).

---

## H1 — Le ratio de couches softmax suffit à combler l'écart de qualité
- **Énoncé** : dans un modèle à 2 couches où seule la couche 1 est softmax (`lin,smax`), la perte val minimale rattrape **au moins 50 %** de l'écart entre tout-linéaire (2.2349) et tout-softmax (2.1235) — soit val ≤ 2.179 — à paramètres égaux (240 480).
- **Critère** : val_min(lin,smax) ≤ 2.179 → confirmée ; sinon → infirmée.
- **Statut** : ✅ **confirmée (moyenne 3 seeds, C2)** — mean(val_min)(lin,smax) = 2.1652 : rattrapage de **80 %** de l'écart (2.2480 → 2.1448). Résidu de 0.103 nat vs softmax, robuste (3/3 seeds). ATTENTION : variance de l'hybride élevée (std 0.058) — le placement du ratio à 2 couches est sensible au seed ; conclusions fines gelées sans ≥ 5 seeds (H0).
- **Conséquence** : 50 % de couches softmax récupèrent ~80 % de l'écart de qualité avec une croissance de KV-cache ÷2. Ouvre la carte des ratios à 4 couches et C3 (delta rule pour le résidu).

## H2 — La delta rule récupère la précision de récupération de l'état
- **Énoncé** : remplacer **uniquement** la mise à jour de l'état `S += k⊗v` par la delta rule `S += β·k⊗(v − S·k)` (β appris par tête via sigmoid(γ), normalisateur Z et feature map φ inchangés — ablation d'une seule chose) récupère **au moins 50 %** de l'écart entre tout-linéaire et tout-softmax, à protocole C2.
- **Critère (calibré sur les moyennes C2, annoncé avant l'expérience)** : mean(val_min)(`dlind,dlind`) ≤ 2.2480 − 0.5×(2.2480 − 2.1448) = **2.1964** → confirmée ; sinon → infirmée. Rattrapage rapporté : (2.2480 − mean_dlind)/0.1032.
- **Secondaire H4b (la régularisation survit-elle ?)** : mean(val@1800)(`dlind,dlind`) ≤ mean(val@1800)(`smax,smax`) − 0.2 = **2.914** → confirmée (l'état delta borne toujours la mémorisation) ; sinon → infirmée.
- **Coût** : +8 paramètres (γ par tête : 2 couches × 4) sur 240 480 (+0.003 %). Entraînement en forme séquentielle + BPTT (coût d'implémentation NumPy, pas conceptuel — la forme parallèle chunked de Yang et al. 2024 est hors périmètre C3).
- **Correction de stabilité (annoncée avant les runs)** : le premier garde-fou a montré que la delta rule **diverge avec des clés non normalisées** (β·‖φ(k)‖² ≫ 1 → état oscillant exponentiellement, gradients ~1e10). Conformément à Schlag et al., les clés sont normalisées après φ : **k̂ = φ(k)/‖φ(k)‖**, utilisées dans l'état, Z et la récupération ; facteur d'écriture (1−β) ∈ [0,1) → stabilité garantie. Cette normalisation fait partie de la définition de `dlind` (elle aussi présent dans la littérature de référence).
- **Statut** : ✅ **confirmée (3 seeds, C3)** — mean(val_min)(`dlind,dlind`) = **1.8976 ± 0.0131** ≤ 2.1964. Au-delà du critère : le delta rule **dépasse le softmax** de 0.247 nat (rattrapage de 340 % de l'écart lin↔smax), 3/3 seeds, variance la plus faible des trois configurations. Voir `RESEARCH_LOG.md` cycle C3.
- **H4b** : ✅ **confirmée** — mean(val@1800)(`dlind,dlind`) = **2.0281** ≤ 2.914. Pente de surapprentissage val@1800 − val_min : **+0.13** (dlind) vs +0.54 (lin) vs +0.97 (smax) → l'état qui remplace borne la mémorisation plus fort que l'état qui additionne ; H4 renforcée.

## H3 — L'entraînement dans la contrainte ternaire vaut mieux que la quantification a posteriori
- **Énoncé** : ré-entraîner avec poids contraints {-1, 0, +1} (surrogate gradient) dégrade la perte val **de moins de 0.15 nat** par rapport au float32, contre une dégradation > 0.30 nat pour la quantification directe du checkpoint.
- **Critère** : Δ(entrain-within) < Δ(quantify-after) / 2 → confirmée.
- **Statut** : ✅ **confirmée (3 seeds, C4)** — **Δ_A = +0.0178 nat** (entraînement dans la contrainte : val_min 1.9154 ± 0.0125 vs float 1.8976) — **sous le seuil de bruit H0** (0.0262) : indistinguable du float. **Δ_B = +2.0853 nat** (quantification a posteriori : val ≈ 4.14–4.19 ≈ hasard, ln 77 = 4.34). Les deux attentes secondaires ✔ (Δ_A < 0.15, Δ_B > 0.30). Bonus régulariseur : pente de surapprentissage +0.019 (vs +0.13 float dlind). Mémoire : poids **939 → 129 Ko (×7.3)**, matmuls ×16.2. La ternarité doit être DANS l'entraînement — confirmation BitNet sur notre stack. Voir `RESEARCH_LOG.md` cycle C4.

## H0 — (méthodologique, permanente) Tout écart < seuil de bruit est ininterprétable sans seeds
- **Énoncé** : à ce scale (240 K, corpus 11 Ko), l'écart type inter-seeds de val_min est attendu dans [0.02, 0.04] nat. Toute comparaison entre configurations dont l'écart moyen est inférieur à 2× cet écart type n'est pas interprétable.
- **Critère de mesure (C2)** : si std_max(configs) < 0.015 → le seuil d'interprétation est abaissé à 2×std_max ; si std_max > 0.05 → plus de seeds requis avant toute conclusion.
- **Statut** : 🔎 **en cours de mesure** (C2 : 3 seeds × 3 configurations) — prérequis à toute conclusion fine.

## H1b — (consolidation C2) La supériorité de l'hybride survit au bruit de seed
- **Énoncé** : l'observation C1 (val_min(lin,smax) < val_min(smax,smax)) est robuste si, **en moyenne sur 3 seeds** {1337, 2024, 777}, mean(val_min(lin,smax)) ≤ mean(val_min(smax,smax)).
- **Critère** : moyennes ordonnées ainsi → H1b confirmée (« établie à 3 seeds ») ; sinon → H1b infirmée (le résultat C1 était un hasard de seed) et le statut de H1 retombe à « indication ».
- **Statut** : ❌ **infirmée (C2)** — mean(lin,smax) = 2.1652 > mean(smax,smax) = 2.1448 ; victoires appariées 1/3. Le dépassement du softmax par l'hybride (C1) était un hasard de seed. Le rattrapage de 80 % (H1) reste valable. Voir `RESEARCH_LOG.md` cycle C2.

## H4 — L'état récurrent agit comme régulariseur (ouverte en C2)
- **Énoncé** : au régime sur-appris (1 800 pas, val_min depuis longtemps dépassé), la val finale de l'hybride est **meilleure** que celle du tout-softmax, parce que la couche à état borne la capacité de mémorisation.
- **Critère** : val@1800(lin,smax) < val@1800(smax,smax) sur ≥ 3 seeds, écart moyen ≥ 0.2 nat.
- **Statut** : ✅ **confirmée à 3 seeds (C2)** — val@1800 : hybride {2.4596, 2.5857, 2.6091} (moy. 2.5515) vs softmax {3.1385, 3.1218, 3.0822} (moy. 3.1142) ; écart moyen **0.56 nat**, 3/3 victoires. **Ré-évaluée en C3 : renforcée** — la delta rule (état qui remplace) sur-apprend encore moins : +0.13 nat de val_min à val@1800, contre +0.54 (lin) et +0.97 (smax).

## H5 — La supériorité de la delta rule survit à la profondeur (C5)
- **Énoncé** : à 4 couches (≈ 464 K params, largeur inchangée d=96) et budget d'entraînement identique, la configuration tout-delta-rule garde l'avantage sur le tout-softmax observé à 2 couches (C3 : 1.8976 vs 2.1448).
- **Critère (annoncé avant l'expérience)** : mean(val_min)(`dlind`×4) ≤ mean(val_min)(`smax`×4) sur 3 seeds → confirmée ; si l'écart < 2×std_max(C5) → « confirmée mais non distinguable du bruit » ; sinon → infirmée.
- **Périmètre budget** : les comparaisons C5 sont INTERNES au cycle (1 200 pas, tous configs identiques) — pas de comparaison directe aux baselines C3/C4 (1 800 pas), consigné.

## H6 — La ternarité reste quasi gratuite à la profondeur (C5)
- **Énoncé** : le surcoût de la contrainte {-1, 0, +1} (C4 : +0.0178 nat à 2 couches) ne s'amplifie pas avec la profondeur.
- **Critère** : mean(val_min)(`dlind`×4 ternaire) − mean(val_min)(`dlind`×4 float) ≤ **0.05 nat** → confirmée ; sinon → coût notable à depth.

## H4c — Le régulariseur survit à la profondeur (C5)
- **Énoncé** : l'effet H4/H4b (l'état borne la mémorisation) persiste quand la capacité double.
- **Critère** : mean(val@1200)(`dlind`×4 float) ≤ mean(val@1200)(`smax`×4) − **0.2 nat** → confirmée ; sinon → infirmée.
- **Statuts** : à tester (C5, seeds {1337, 2024, 777}, protocole commité avant les runs).
