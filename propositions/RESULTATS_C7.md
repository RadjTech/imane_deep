# Résultats C7 — validation sur corpus réel fr1m (2026-11-24)

> **AMENDEMENT n°2 (25/11/2026) — chiffres « œuvres hors corpus » remplacés.**
> `data/test_oeuvres.txt` n'avait pas été nettoyé comme le corpus : 13 294 caractères hors
> alphabet (dont 13 154 apostrophes courbes ’) étaient comptés comme ¤. Les valeurs
> `test_oeuvres` du tableau §1 et la mention « 2.0323 » en §3-5 étaient **invalides** (défaut
> 1 documenté par le relecteur). Réparation : `data/test_oeuvres_v2.txt` (140 ¤ / 1 234 422 =
> 0.0113 %), les **mêmes 25 checkpoints** réévalués (aucun run relancé, même commit `43f9144`).
>
> **test_oeuvres_v2 (poids finaux, 5 seeds)** — 140 caractères inconnus partout :
>
> | variante | moyenne ± écart type |
> |---|---|
> | `smax` | **1.8830 ± 0.0063** |
> | `dlind-elu1` | 1.8967 ± 0.0047 |
> | `dlind-hist` | 1.9044 ± 0.0045 |
> | `dlind-const` | 1.9086 ± 0.0045 |
> | `lin-hist` | 2.1864 ± 0.0054 |
>
> L'ordre des variantes est **inchangé** par rapport au test principal (smax < dlind-elu1 <
> dlind-hist < dlind-const < lin-hist) : les verdicts C7-V1…V4 ne sont pas affectés.
> Tout chiffre « œuvres » antérieur à cet amendement est retiré.

> **AMENDEMENT n°3 (25/11/2026) — correction d'une phrase (§3 point 4).**
> J'avais écrit « tous les modèles sont en sous-apprentissage : val@1800 < val_min » —
> cela compare **deux mesures différentes** (val tirée en 10 batchs vs val entière
> déterministe). Correction : sur la **même** mesure (val entière), les **25/25 runs
> finissent à leur minimum au pas 1 800** (`val_entiere_min == val_entiere_final`) — donc
> 1 800 pas est trop court (confirmé par la courbe à pas égaux du relecteur : dlind devant
> smax jusqu'au pas ~1 250, smax devant ensuite). La conclusion reste : le budget borne la
> comparaison — c'est exactement la question du cycle C8.

> Protocole `PROTOCOLE_C7.md` commité **avant** tout run (`43f9144`, avec `analyse_c7.py`,
> `c7_verification.py` et l'évaluateur `evaluer_test.py` — contrôle d'équivalence avec
> `evaluer_fixe` : écart 0.0, 1 117 prédictions). Runs : `scripts/lancer_parallele.py`,
> cycle `c7a`, commit unique `43f9144`, `code_modifie: false`, **25/25 runs, 0 échec**,
> 4 481 s de calcul (2 processus à 1 thread). Test évalué **une seule fois**, en fin,
> sur les poids finaux. Vérification indépendante (quantile de Student recalculé par
> intégration, pas de table) : **IDENTIQUE** à 4.9e-05 près — après correction d'un bug
> de ma fonction quantile (CDF intégrée de 0 à t au lieu de −∞ à t) détecté par l'écart,
> corrigé avant toute conclusion (amendement n°1, daté ci-dessous).

## Amendement n°1 (2026-11-24, avant l'analyse)

1. `c7_verification.py` attendait 19 lignes dans `historique.csv` ; le fichier en a 20
   (en-tête + 19 points). Corrigé en 20 — aucun effet sur les résultats.
2. Le quantile de Student de `c7_verification.py` intégrait la densité de 0 à t au lieu
   de −∞ à t → IC sans rapport. Corrigé (`0.5 + ∫₀ᵗ`), quantile recalculé 2.7764 vs
   table 2.776, vérification rejouée → IDENTIQUE. Les verdicts ci-dessous n'ont été
   lus qu'après cette correction.

## 1. Résultats (perte sur le TEST, poids finaux, 5 seeds {101, 202, 303, 404, 505})

| Variante | test (5 seeds) | moyenne ± écart type | test_oeuvres (2 auteurs hors corpus) | val entière min | val@1800 |
|---|---|---|---|---|---|
| `smax` | 1.8299 1.8267 1.8185 1.8251 1.8115 | **1.8223 ± 0.0074** | **2.0323** | 1.8166 | 1.8062 |
| `dlind-elu1` | 1.8464 1.8463 1.8302 1.8405 1.8328 | 1.8392 ± 0.0075 | 2.0424 | 1.8382 | 1.8320 |
| `dlind-hist` | 1.8582 1.8465 1.8399 1.8441 1.8444 | 1.8466 ± 0.0069 | 2.0497 | 1.8459 | 1.8398 |
| `dlind-const` | 1.8599 1.8489 1.8455 1.8445 1.8486 | 1.8495 ± 0.0061 | 2.0535 | 1.8490 | 1.8428 |
| `lin-hist` | 2.1640 2.1677 2.1527 2.1586 2.1624 | 2.1611 ± 0.0057 | 2.3099 | 2.1566 | 2.1521 |

Référence non neuronale (même val, même protocole) : n-gramme Kneser-Ney 7-grammes
**1.3402** (`results/c7_ngram_fr1m.json`, 5.4 s d'entraînement, 297 Mo).

Caractères de test remplacés par ¤ : **0** (les corpus ont déjà le bon alphabet) ;
1 117 caractères de val pour le contrôle d'équivalence, conformes au chiffre C3b.

## 2. Verdicts (critères pré-enregistrés, différences appariées par seed)

| # | comparaison | différence | IC 95 % | verdict |
|---|---|---|---|---|
| **C7-V1** | `dlind-elu1` − `smax` (test) | **+0.0169** | [+0.0122 ; +0.0216] | **INFIRMÉE** — `smax` fait mieux, 5/5 seeds, les distributions ne se chevauchent pas |
| **C7-V2** | `dlind-const` − `dlind-elu1` (test) | +0.0102 | [+0.0023 ; +0.0182] | **CONFIRMÉE** — borne haute +0.018 ≤ +0.05 : la version simple suffit (à +0.01 près) |
| **C7-V3** | `dlind-elu1` − `dlind-hist` (test) | **−0.0074** | [−0.0138 ; −0.0009] | **la φ corrigée est meilleure** — l'inverse du petit corpus (C3b-H6 : +0.095) |
| **C7-V4** | meilleur réseau (val) vs n-gramme | 1.8166 vs 1.3402 | — | **descriptif : aucun réseau ne bat le n-gramme** (marge 0.48 nat) |

`test_oeuvres` (œuvres entières d'auteurs absents du corpus) donne le même ordre :
`smax` 2.0323 < `dlind-elu1` 2.0424 < `dlind-hist` 2.0497 < `dlind-const` 2.0535 < `lin-hist` 2.3099.

## 3. Lecture honnête

1. **Le résultat central de C3b ne se généralise pas** : à 11 Ko, `dlind` battait `smax`
   de 0.165 (5/5) ; à 1 Mo, `smax` bat `dlind` de 0.017 (5/5, IC sans chevauchement).
   Les mesures sur petit corpus reflétaient surtout la **régularisation** (verrou déjà
   identifié par le n-gramme 1.683), pas la qualité du mécanisme pour une langue.
2. **Mais l'essentiel de C3b tient** : `lin` reste très loin derrière (2.161 — l'écart
   `lin`→`dlind` vaut 0.32, quinze fois l'écart `dlind`→`smax`), et `dlind-const` (sans
   clé ni requête) suit `dlind-elu1` à +0.01 : le gros du gain vient de l'**oubli borné**,
   pas de l'adressage par contenu — confirmé à 1 Mo.
3. **La φ corrigée redevient la meilleure** avec de vraies données (V3) : à 11 Ko, la φ
   historique faisait office de régulariseur gratuit ; à 1 Mo, elle coûte 0.007.
4. **Tout le monde est en sous-apprentissage** : val@1800 < val_min pour les 5 variantes
   (ex. smax 1.8062 < 1.8166) — à 1 800 pas, aucun modèle n'a fini de progresser. Le
   budget du protocole (identique C3b) borne la comparaison ; le classement au régime
   final n'est pas mesuré.
5. **Le n-gramme reste champion** : 3.82 lettres d'hésitation contre 6.2 pour le meilleur
   réseau. À 240 K paramètres et 1 800 pas, un compteur de caractères à 7 contextes bat
   tous nos réseaux sur la val comme sur le test — la priorité reste les données et la
   capacité, pas uniquement le mécanisme.

## 4. Limites consignées

- Corpus fr1m uniquement (998 K car.) ; fr5m préparé mais **non lancé** : l'évaluation val
  y coûte 4× plus par point (474 K caractères) — nécessite l'option `--pas-eval` prévue
  par la mission et un accord de durée séparé.
- Un seul passage du test (protocole) : les chiffres du test ont une incertitude
  d'échantillonnage propre, non estimée (par construction, aucune re-lecture).
- 5 seeds (minimum du protocole) ; les écarts entre `dlind-*` (0.007–0.010) sont petits
  devant l'écart `lin` (0.32) mais les IC correspondants sont étroits (±0.003-0.006).
- Machine de mesure : 2 cœurs, Python 3.13.18, NumPy 2.3.5 ; rejeu exact vérifié en C3b,
  pas re-vérifié ici (runs d'une autre famille).

## 5. Suite recommandée

1. **fr5m** (avec `--pas-eval`) : tester la tendance avec 5× plus de données.
2. **Budget d'entraînement** : 1 800 pas est un plafond artificiel ici (sous-apprentissage
   général) — prolonger les runs avant de conclure sur le mécanisme.
3. **Capacité** : le n-gramme à 1.34 impose un plancher — tester d (96 → 192) ou 4 couches
   sur fr1m si la question « réseau vs compteur » doit être tranchée.
