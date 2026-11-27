# Résultats C11-bis — le bon budget sur les bonnes données (fr5m_v2, 18 000 pas) (2026-11-27)

> Protocole `PROTOCOLE_C11B.md` commité avant tout run (`c6a4d9c`) ; exécution GitHub
> Actions (même mécanisme que C8/C11) ; 9/9 runs intègres (ENV `cycle: c11b`, corpus
> `fr5m_v2`, budget 18 000, code non modifié ; historique 32 lignes = en-tête + 31 points ;
> les `checkpoint*.npz` sont exclus du dépôt par le workflow — par conception). Réparation
> préalable : des fichiers locaux tronqués par une restauration du bac à sable ont été
> re-committés (`be0a5bf`). Vérification indépendante `scripts/verif_c11b.py` :
> **IDENTIQUE** (4.4e-05). N-gramme de référence : celui de C11 (1.222, mesuré avant
> tout réseau, job séparé).

## 1. Tableau (test_best — poids choisis par la val, une seule éval du test)

| Variante | test_best (3 seeds) | moyenne ± écart type | val entière | œuvres_v2 |
|---|---|---|---|---|
| `smax` | 1.2981 1.3028 1.3016 | **1.3008 ± 0.0024** | **1.2886** | **1.4587** |
| `dlind-elu1` | 1.3601 1.3613 1.3551 | 1.3588 ± 0.0033 | 1.3468 | 1.5055 |
| `dlind-const` | 1.4406 1.4447 1.4382 | 1.4412 ± 0.0033 | 1.4303 | 1.5829 |

Compteur de lettres (val fr5m_v2) : 1.222. Tous les pas_best = 18 000 : les runs finissent
à leur minimum, **encore du gain disponible**.

## 2. Verdicts pré-enregistrés

| # | Comparaison | Valeur | Verdict |
|---|---|---|---|
| **C11B-V1** (descriptif) | écart meilleur réseau ↔ compteur sur la val | **+0.0666** (1.2886 vs 1.2220) — contre **+0.2517** à b5400 | **le budget referme le trou** (÷3.8 quand le budget ×3.3) |
| **C11B-V2** | dlind-elu1 − smax @18k | +0.0580 [+0.0474 ; +0.0686] | **infirmée** — la classique gagne encore |
| **C11B-V3** | dlind-const − dlind-elu1 ≤ +0.05 | +0.0824 [+0.0784 ; +0.0863] | **infirmée** |
| **C11B-V4** (descriptif) | test_best b18000 vs b5400 | smax −0.184 · dlind-elu1 −0.217 · dlind-const −0.182 | tout le monde gagne du budget |

## 3. Lecture honnête

1. **La trajectoire est claire et cohérente entre corpus** : l'écart réseau↔compteur
   tombe d'un facteur ~4 quand le budget ×3.3, sur les DEUX corpus (fr1m : 0.148 → 0.006 ;
   fr5m_v2 : 0.252 → 0.067). Le trou avec le compteur est une question de **budget**, pas
   de plafond. Prolonger (≈ 60 000 pas, ~1,7 h/run sur GitHub, seeds +1) devrait le fermer ;
   non mesuré (protocole à écrire avant).
2. **Le duel des mécanismes ne bascule pas** : smax devant de +0.058 (IC étroit) — écart
   un peu plus petit qu'à b5400 (+0.091) mais loin de se fermer ; `dlind-const` s'écarte
   davantage au budget long (+0.082), comme sur fr1m (+0.065). L'adressage par contenu
   sert de plus en plus quand l'entraînement s'allonge.
3. **En hésitation de lettres (exp de la perte, par corpus)** : sur fr5m_v2, le meilleur
   réseau hésite entre **3,67** lettres, le compteur entre **3,39** — sur fr1m, 3,91 vs
   3,80 (C8). Les deux corpus racontent la même histoire : proche, mais derrière, à ce
   budget ; la pente favorise le réseau.
4. **Contre-hypothèses** : pas de fuite (fr5m_v2 conforme 0 copie) ; même protocole entre
   variantes ; test lu une fois ; seeds neufs ; ordinateurs GitHub différents d'un run à
   l'autre (pertes comparables, temps non utilisées).
5. **Coût/qualité (rappel C9/C10)** : `dlind-const` reste 6× plus rapide à long contexte
   avec 0,8 Ko d'état ; à ce régime son écart de qualité vaut +0.14 test (vs smax).

## 4. Suites (dans l'ordre des leviers)

1. **fr5m_v2, 60 000 pas** (budget ×3.3 encore) — devrait fermer le trou (pente mesurée
   ×4). Coût : ~1,7 h/run × 3 variantes × 3 seeds, en parallèle ≈ 2 h sur GitHub.
2. Si l'écart stagne malgré le budget : capacité d = 192 (protocole à commité avant).
3. Le duel des mécanismes est clos en l'état : smax pour la qualité, dlind-const pour le
   coût à long contexte — à arbitrer selon l'usage.
