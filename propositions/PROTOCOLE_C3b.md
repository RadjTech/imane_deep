# Cycle C3b — Re-mesurer les fondations (brouillon de pré-enregistrement)

> **Statut : brouillon du 24/11/2026, rien n'est lancé** (pilotes exploratoires seulement). À relire, modifier, puis commiter AVANT les runs
> (protocole, étape 3). Les pilotes du 23/11/2026 (1 à 3 seeds, dossier `pilotes/`) ont servi
> à choisir les questions et les seuils : ils sont **exploratoires** et ne comptent pas comme
> confirmation. La confirmation se fait sur des seeds **neufs**.

## 1. Questions
1. Les conclusions de C1–C3 survivent-elles à une φ conforme (ELU+1) ?
   Aujourd'hui `phi` (hybrid_numpy.py:57) vaut exp(x)+1 pour x ≤ 0 : saut de 2 à 1 en zéro, φ ≥ 1 partout.
2. D'où vient le gain de `dlind` sur `lin` ? C3 a changé **trois** choses à la fois :
   la normalisation des clés, l'oubli (β < 1) et l'adressage par contenu.

## 2. Hypothèses (critère numérique unique par hypothèse)

Toutes les comparaisons sont **appariées par seed**, métrique principale = minimum de la
perte sur la **val entière** (`val_entiere_min`, déterministe), IC à 95 % de Student.

| # | Énoncé | Confirmée si |
|---|---|---|
| H5 | L'oubli exponentiel suffit : sans clé ni requête, la qualité est la même que celle de `dlind` | IC 95 % de (`dlind-const` − `dlind-hist`) ⊂ [−0,05 ; +0,05] |
| H6 | La φ du dépôt régularise `dlind` (la φ correcte est moins bonne) | IC 95 % de (`dlind-elu1` − `dlind-hist`) entièrement > 0 |
| H7 | `dlind` reste meilleur que `smax` avec la φ correcte | IC 95 % de (`dlind-elu1` − `smax`) entièrement < 0 |
| H9 | H1 survit à φ corrigée : l'hybride rattrape au moins la moitié de l'écart `lin` → `smax` | moyenne appariée de (`lin-elu1` − `hybride-elu1`) ≥ 0,5 × moyenne de (`lin-elu1` − `smax`), **et** cette dernière a un IC 95 % > 0 |
| H8 | La normalisation des clés seule n'explique pas le gain de C3 | IC 95 % de (`lin-hist-nk` − `lin-hist`) ⊂ [−0,05 ; +0,05] |

Si l'IC ne tranche pas (ni confirmée ni infirmée), statut **indéterminée** et passage à 8 seeds.

## 3. Protocole
- Variantes : `smax`, `hybride-hist`, `hybride-elu1`, `lin-hist`, `lin-elu1`, `lin-hist-nk`, `dlind-hist`, `dlind-elu1`, `dlind-const`.
- Seeds **neufs** (jamais utilisés en C1–C3 ni en pilote) : 11, 22, 33, 44, 55.
- 1 800 pas, tout le reste identique à C2/C3. 45 runs.
- Référence non neuronale : n-gramme Kneser-Ney (`ngram_baseline.py`), **1,683 nat** sur le corpus actuel.

```
python3 scripts/c3b_controles.py --check
python3 scripts/gardefou_conformite.py
python3 scripts/lancer_parallele.py --cycle c3b --variantes \
    --configs smax hybride-hist hybride-elu1 lin-hist lin-elu1 lin-hist-nk dlind-hist dlind-elu1 dlind-const \
    --seeds 11 22 33 44 55
python3 scripts/agreger.py --prefixes c3b --ref dlind-hist --metrique val_entiere_min --json results/c3b/analyse.json
```

`gardefou_conformite.py` teste explicitement la φ « elu1 » (celle des variantes `-elu1`) et passe.
Les variantes `-hist` utilisent la φ historique, qui reste le défaut du module (témoins de C1–C5).

## 4. Garde-fous
Forme rapide = code du dépôt (perte 0, gradients < 2e-15) et différences finies < 1e-4 pour
toutes les variantes (`--check`).

## 5. Ce que chaque issue change
| Issue | Conséquence |
|---|---|
| H5 confirmée | La couche de référence devient l'oubli exponentiel : état de 96 nombres au lieu de 2 400 par couche, q et k inutiles. Le titre de C3 (« précision de récupération ») est à reformuler |
| H5 infirmée | L'adressage par contenu compte : garder `dlind` |
| H6 confirmée | H4/H4b (régularisation par l'état) sont à ré-attribuer en partie à φ |
| H7 infirmée | Le résultat phare de C3 tombe |

## 6. Limite assumée
À 11 Ko, un n-gramme bat tous les modèles : C3b dit quelles affirmations passées tiennent, pas
quel mécanisme est le meilleur pour une langue. Cette question demande un corpus de plusieurs Mo
avec un jeu de test (`preparer_corpus.py`).
