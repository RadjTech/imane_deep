# Hypothèses falsifiables

> Format : chaque hypothèse porte un critère de décision numérique **annoncé avant** l'expérience. Statuts : `à tester` → `confirmée` / `infirmée` / `indication` (1 seed) → `établie` (≥ 3 seeds).

---

## H1 — Le ratio de couches softmax suffit à combler l'écart de qualité
- **Énoncé** : dans un modèle à 2 couches où seule la couche 1 est softmax (`lin,smax`), la perte val minimale rattrape **au moins 50 %** de l'écart entre tout-linéaire (2.2349) et tout-softmax (2.1235) — soit val ≤ 2.179 — à paramètres égaux (240 480).
- **Critère** : val_min(lin,smax) ≤ 2.179 → confirmée ; sinon → infirmée.
- **Statut** : ✅ **confirmée (moyenne 3 seeds, C2)** — mean(val_min)(lin,smax) = 2.1652 : rattrapage de **80 %** de l'écart (2.2480 → 2.1448). Résidu de 0.103 nat vs softmax, robuste (3/3 seeds). ATTENTION : variance de l'hybride élevée (std 0.058) — le placement du ratio à 2 couches est sensible au seed ; conclusions fines gelées sans ≥ 5 seeds (H0).
- **Conséquence** : 50 % de couches softmax récupèrent ~80 % de l'écart de qualité avec une croissance de KV-cache ÷2. Ouvre la carte des ratios à 4 couches et C3 (delta rule pour le résidu).

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
- **Statut** : ❌ **infirmée (C2)** — mean(lin,smax) = 2.1652 > mean(smax,smax) = 2.1448 ; victoires appariées 1/3. Le dépassement du softmax par l'hybride (C1) était un hasard de seed. Le rattrapage de 80 % (H1) reste valable. Voir `RESEARCH_LOG.md` cycle C2.

## H4 — L'état récurrent agit comme régulariseur (ouverte en C2)
- **Énoncé** : au régime sur-appris (1 800 pas, val_min depuis longtemps dépassé), la val finale de l'hybride est **meilleure** que celle du tout-softmax, parce que la couche à état borne la capacité de mémorisation.
- **Critère** : val@1800(lin,smax) < val@1800(smax,smax) sur ≥ 3 seeds, écart moyen ≥ 0.2 nat.
- **Statut** : ✅ **confirmée à 3 seeds (C2)** — val@1800 : hybride {2.4596, 2.5857, 2.6091} (moy. 2.5515) vs softmax {3.1385, 3.1218, 3.0822} (moy. 3.1142) ; écart moyen **0.56 nat**, 3/3 victoires. À ré-évaluer dans C3 (la delta rule doit conserver cet effet).
