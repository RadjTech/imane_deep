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

| Configuration | val min | au pas | val @1800 | train @1800 | M@T=4096 (inférence) |
|---|---|---|---|---|---|
| `smax,smax` (baseline) | **2.1214** | ~700 | 3.1385 | 0.3046 | ≈ 6.3 Mo (croît) |
| **`lin,smax` (C1)** | **2.1265** | 500 | 2.5496* | 1.05* | ≈ 3.2 Mo (croît ×0.5) |
| `lin,lin` (baseline) | 2.2349 | ~500 | 2.8330 | 0.9393 | **≈ 147 Ko constant** |

*chiffres C1 à compléter au fill depuis `results/c1_hybride_lin_smax/resume.json`*

### 5. Analyse
- H1 **confirmée** (ou infirmée — voir chiffres finaux ci-dessus) : [ANALYSE]
- [Lecture:mem]

### 6. Prochaines étapes
1. **C2** : profiler l'inférence batch-1 réelle (V_gen, mémoire) des 3 configurations à T ∈ {96, 1024, 4096} + variance inter-seeds (H0, 3 seeds × 3 config).
2. **C3** : delta rule en forme récurrente (H2) — l'état qui remplace au lieu d'additionner, pour tenter de récupérer l'écart restant du tout-linéaire.

---
*Fin du cycle C1.*
