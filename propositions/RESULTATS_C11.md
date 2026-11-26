# Résultats C11 — plus de données (fr5m_v2, 5.0 M caractères) (2026-11-26)

> Protocole `PROTOCOLE_C11.md` (commité `3754d96`) + **amendement n°1** du prédécesseur
> (`2d3e094`, commité avant tout run) : exécution sur **GitHub Actions** (9 runs en
> parallèle, 1 thread chacun), protocole inchangé ; le n-gramme est mesuré dans un job
> séparé en parallèle des runs (justifié : il ne lit aucun résultat des réseaux).
> 9/9 runs intègres (ENV `cycle: c11`, corpus `fr5m_v2`, code non modifié).
> Vérification indépendante `scripts/verif_c11.py` : **IDENTIQUE** (2.9e-05).
> Aucun IC indéterminé → pas d'amendement seeds.

## 1. Tableau (test_best — poids choisis par la val, une seule éval du test)

| Variante | test_best (3 seeds) | moyenne ± écart type | val entière | œuvres_v2 |
|---|---|---|---|---|
| `smax` | 1.4826 1.4825 1.4896 | **1.4849 ± 0.0040** | **1.4737** | **1.6104** |
| `dlind-elu1` | 1.5776 1.5724 1.5775 | 1.5758 ± 0.0030 | 1.5663 | 1.6822 |
| `dlind-const` | 1.6205 1.6249 1.6240 | 1.6231 ± 0.0023 | 1.6138 | 1.7251 |

Référence compteur de lettres (val entière de fr5m_v2) : **1.222** (elle profite des
données : 1.3402 sur fr1m → 1.222 sur fr5m_v2).

## 2. Verdicts pré-enregistrés

| # | Comparaison | Différence | IC 95 % | Verdict |
|---|---|---|---|---|
| **C11-V1** | dlind-elu1 − smax @fr5m_v2/b5400 | **+0.0909** | [+0.0818 ; +0.1000] | **infirmée** — la classique gagne |
| **C11-V2** | dlind-const − dlind-elu1 ≤ +0.05 | +0.0473 | [+0.0352 ; +0.0594] | **infirmée** (borne +0.0594 > +0.05) |
| **C11-V3** (descriptif) | val — meilleur réseau vs n-gramme fr5m_v2 | 1.4737 vs 1.222 | écart +0.2517 | le réseau est derrière le compteur à ce budget |
| **C11-V4** (descriptif) | IC fr5m_v2 vs écart C8-fr1m/b5400 [+0.0787 ; +0.0950] | — | chevauchement ✔ | **compatible : plus de données ne change pas le duel des mécanismes** |

## 3. Lecture honnête

1. **Le duel smax / dlind ne dépend pas de la quantité de données à budget égal** :
   +0.0909 sur fr5m_v2 contre +0.0868 sur fr1m (IC chevauchants, V4). smax reste devant,
   5/5 puis 3/3 seeds du même côté, IC étroits. Résultat négatif consigné.
2. **L'écart réseau↔compteur AUGMENTE à budget égal** : 0.148 (fr1m/b5400) → 0.252
   (fr5m_v2/b5400). Pourquoi : à 5 400 pas × 1 536 caractères, le réseau voit ~1.8 époques
   de fr5m_v2 (contre ~9.7 de fr1m) — il n'a pas le temps d'exploiter les données
   supplémentaires (+0.014 seulement), tandis que le compteur, qui n'a pas de budget de
   « pas », absorbe tout le texte (−0.118). **Le budget doit croître avec les données.**
   C8 l'a montré : à 18 000 pas sur fr1m, l'écart tombait à 0.006. La combinaison
   fr5m_v2 + 18 000 pas (≈ 3× le coût de C8-18k) n'est pas mesurée — c'est le prochain
   levier naturel, pas la capacité (le tableau de décision ne la déclenche pas).
3. **La version simple reste limite** : +0.0473, borne haute +0.0594 — au-dessus du seuil
   +0.05 à ce budget également (comme C8-18k : +0.0646). Constat cohérent : plus
   l'entraînement est long ou riche, plus l'adressage par contenu (Q/K) sert.
4. **Contre-hypothèses** : fuite exclue (fr5m_v2 conforme, 0 copie) ; même protocole
   d'éval entre variantes ; test lu une fois ; ordinateurs GitHub différents d'un run à
   l'autre — les pertes restent comparables (C8 l'a établi sur 45 runs), les temps non
   (non utilisées ici).
5. **Ce que C11 ne dit pas** : le comportement à budget proportionné aux données
   (fr5m_v2, 18 000+ pas), et le point de croisement coût/qualité (C9 : dlind-const 6×
   plus rapide à long contexte, 0.8 Ko d'état — à qualité +0.10-0.16 nat derrière smax
   selon la variante).

## 4. État des critères de la mission après C11

| Tableau de décision | Constat | Action |
|---|---|---|
| écart réseau↔n-gramme < 0.3 → données | fait (C11) | fait — le levier suivant est le **budget** (fr5m_v2, 18 000 pas) |
| dlind-elu1/const devient meilleure quelque part | non observé (C7, C8, C10-V1 indéterminé, C11) | — |
| smax gagne en qualité, dlind-const bien moins chère | confirmé (C9/C10/C11) | le dire clairement au bilan final |
| IC ne tranche pas | C10-V1 → amendement seeds en cours (744/755/766) | conclusion C10 après les runs |
