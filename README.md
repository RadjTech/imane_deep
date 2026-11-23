# imane_deep

> **Mission** : construire une IA de langage avancée qui **ne dépend pas du GPU** — en remplaçant l'attention softmax O(T²) par des mécanismes à état linéaires/constants, en ternarisant le calcul, et en mesurant tout dans les **unités du CPU** (qualité/bit, qualité/(Go/s), qualité/watt).

Recherche menée en **cycles complets et traçables** (méthode « get-research-done » + expériences agentiques : chaque cycle est lancé, mesuré, journalisé et commité de bout en bout).

## La question centrale

> *Quelle est la dépense minimale de « précision softmax » (nombre de couches, bits par poids, octets d'état) pour conserver la qualité d'un transformer, tout en rendant l'entraînement et surtout l'inférence viables sur CPU seul ?*

## L'état du dépôt

| Cycle | Question | Hypothèse | Statut |
|---|---|---|---|
| **C1** | Combien de couches softmax suffisent ? (0/2, 1/2, 2/2) | H1 | ✅ terminé — voir `RESEARCH_LOG.md` |
| C2 | Bruit de seed (3×3) + profilage d'inférence batch-1 | H0, H1b | ✅ terminé — H1b infirmée, H4 ouverte |
| C3 | Delta rule : l'état qui remplace au lieu d'additionner | H2, H4b | ✅ terminé — H2 confirmée (dlind **dépasse** smax : 1.8976 vs 2.1448), H4b confirmée |
| C4 | Ternarisation façon BitNet : entraîner avec {-1, 0, +1} | H3 | ✅ terminé — H3 confirmée (Δ +0.018 nat, sous le bruit ; quantify-after : +2.09 ≈ hasard ; mémoire ×7.3) |
| C5 | Profondeur : les gains survivent-ils à 4 couches ? | H5, H6, H4c | ✅ terminé — H5 confirmée (amplifiée : −0.333 nat), H4c confirmée (−0.70), H6 infirmée de justesse (+0.057, effet budget) |
| C6 | Budget : le coût du ternaire à 4 couches est-il un artefact de budget ? | H6b | à venir |

## Arborescence

```
imane_deep/
├── RESEARCH_PROTOCOL.md   ← LA discipline (les 9 étapes obligatoires d'un cycle)
├── HYPOTHESES.md          ← les hypothèses falsifiables et leur statut
├── RESEARCH_LOG.md        ← le journal : chaque cycle, chiffres et décision
├── METRICS_CPU.md         ← définitions des métriques « CPU-first »
├── experiments/
│   └── hybrid_numpy.py    ← modèle à types par couche (lin / smax), NumPy pur,
│                             forward + backward manuels vérifiés par différences finies
├── results/               ← CSV, résumés JSON, logs d'entraînement (traçabilité)
├── data/corpus_fr.txt     ← corpus d'entraînement (11 172 caractères de français)
└── .github/workflows/     ← CI : garde-fous rejoués à chaque push (GitHub Actions, CPU)
```

## Reproduire

```bash
python3 -m pip install numpy                # seule dépendance
python3 experiments/hybrid_numpy.py --check-grads        # GARDE-FOU obligatoire
python3 experiments/hybrid_numpy.py --check-ternary      # garde-fou C4 (ternaire)
python3 experiments/hybrid_numpy.py --train --types "lin,smax" --out-dir results/c1_hybride_lin_smax
python3 experiments/hybrid_numpy.py --train --types "smax,smax" --out-dir results/c1_baseline_smax
python3 experiments/hybrid_numpy.py --train --types "lin,lin" --out-dir results/c1_baseline_lin
```

Protocole identique pour les trois configurations : 240 480 paramètres, corpus 11 172 caractères (vocab 77), batch 16 × T 96, 1 800 pas, AdamW (warmup 100 → cosinus), seed 1337. Seule variable : le type d'attention par couche — **ablation propre**.

## Contexte scientifique

Le goulot d'une inférence LLM sur CPU est la **bande passante mémoire** (~100 Go/s contre ~1 To/s côté GPU) : chaque bit économisé est de la vitesse gagnée. Trois familles de leviers sont explorées ici, en cohérence avec l'état de l'art : architectures à état (RWKV-7, Mamba-2, GLA — coût séquentiel linéaire, mémoire constante), poids ternaires (BitNet b1.58 — multiplications remplacées par des additions entières, 16-32× de mémoire en moins), et hybrides (Jamba, Granite 4, Zamba-2 — quelques couches précises au milieu de couches récurrentes bon marché).

## Fonctionnement agentique

Les cycles sont exécutés par un agent (Arena Agent Mode) qui suit `RESEARCH_PROTOCOL.md` : garde-fou gradients → expérience → analyse → mise à jour du journal → commit → push. Pour lancer le cycle suivant : demander « cycle suivant » dans la conversation Arena.
