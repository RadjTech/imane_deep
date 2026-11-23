# Hypothèses falsifiables

> Format : chaque hypothèse porte un critère de décision numérique **annoncé avant** l'expérience. Statuts : `à tester` → `confirmée` / `infirmée` / `indication` (1 seed) → `établie` (≥ 3 seeds).

---

## H1 — Le ratio de couches softmax suffit à combler l'écart de qualité
- **Énoncé** : dans un modèle à 2 couches où seule la couche 1 est softmax (`lin,smax`), la perte val minimale rattrape **au moins 50 %** de l'écart entre tout-linéaire (2.2349) et tout-softmax (2.1235) — soit val ≤ 2.179 — à paramètres égaux (240 480).
- **Critère** : val_min(lin,smax) ≤ 2.179 → confirmée ; sinon → infirmée.
- **Statut** : ✅ **confirmée (seed unique)** — val_min(lin,smax) = **2.1112** (pas 700) : rattrapage de 111 % de l'écart, et 0.0123 nat *sous* le tout-softmax. Le dépassement du softmax reste une « indication » (sous le seuil de bruit H0, 1 seed) ; le rattrapage du tout-linéaire est robuste (0.12 nat ≫ bruit). Voir `RESEARCH_LOG.md` cycle C1. À consolider par 3 seeds en C2.
- **Conséquence** : 50 % de couches softmax ≈ la qualité du softmax (voire mieux, à confirmer) avec une croissance de KV-cache divisée par 2. Ouvre C2 (profilage) et la carte des ratios à 4 couches.

## H2 — La delta rule récupère la précision de récupération de l'état
- **Énoncé** : remplacer `S += k⊗v` par `S += β·k⊗(v − Sᵀk)` (l'état écrase la trace similaire au lieu d'additionner) réduit d'au moins 50 % l'écart val entre tout-linéaire et tout-softmax **à coût paramètres égal**.
- **Critère** : val_min(delta) ≤ 2.23 − 0.5×(2.23−2.12) = **2.175** → confirmée.
- **Statut** : à tester (cycle C3 — prérequis : forme récurrente + BPTT, backward vérifié).

## H3 — L'entraînement dans la contrainte ternaire vaut mieux que la quantification a posteriori
- **Énoncé** : ré-entraîner avec poids contraints {-1, 0, +1} (surrogate gradient) dégrade la perte val **de moins de 0.15 nat** par rapport au float32, contre une dégradation > 0.30 nat pour la quantification directe du checkpoint.
- **Critère** : Δ(entrain-within) < Δ(quantify-after) / 2 → confirmée.
- **Statut** : à tester (cycle C4).

## H0 — (méthodologique, permanente) Tout écart < seuil de bruit est ininterprétable sans seeds
- **Énoncé** : à ce scale (240 K, corpus 11 Ko), l'écart type inter-seeds de val_min est attendu dans [0.02, 0.04] nat. Toute comparaison entre configurations dont l'écart moyen est inférieur à 2× cet écart type n'est pas interprétable.
- **Critère de mesure (C2)** : si std_max(configs) < 0.015 → le seuil d'interprétation est abaissé à 2×std_max ; si std_max > 0.05 → plus de seeds requis avant toute conclusion.
- **Statut** : 🔎 **en cours de mesure** (C2 : 3 seeds × 3 configurations) — prérequis à toute conclusion fine.

## H1b — (consolidation C2) La supériorité de l'hybride survit au bruit de seed
- **Énoncé** : l'observation C1 (val_min(lin,smax) < val_min(smax,smax)) est robuste si, **en moyenne sur 3 seeds** {1337, 2024, 777}, mean(val_min(lin,smax)) ≤ mean(val_min(smax,smax)).
- **Critère** : moyennes ordonnées ainsi → H1b confirmée (« établie à 3 seeds ») ; sinon → H1b infirmée (le résultat C1 était un hasard de seed) et le statut de H1 retombe à « indication ».
- **Statut** : à tester (C2, annoncé avant l'expérience).
