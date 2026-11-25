# Résultats C8 — le budget d'entraînement (GitHub Actions, 45/45 runs) (2026-11-25)

> Protocole `PROTOCOLE_C8_GITHUB.md` commité **avant** tout run (`46daced`). Runs sur
> GitHub Actions (ubuntu-latest, 1 thread), commit `46daced`, code non modifié — vérifié
> par `ENV.json`. **45/45 runs intègres** (resume.json, test.json, historique, ENV —
> contrôle local). Verdicts recalculés par un **second script indépendant**
> (`scripts/verif_c8_github.py`, quantile de Student recalculé par intégration) :
> **IDENTIQUE à l'analyse officielle** (`results/c8g_analyse.json`,
> `RESULTATS_C8_brut.md`) au 4e décimal près — `results/c8g_verification.json`.
> Aucun IC indéterminé → pas d'amendement seeds supplémentaires.

## 1. Tableau (métrique principale : perte **test**, poids choisis par la val, une éval)

| Variante | 1 800 pas | 5 400 pas | 18 000 pas | œuvres_v2 @18 000 | val_best @18 000 |
|---|---|---|---|---|---|
| `smax` | 1.8204 ± 0.0097 | 1.4935 ± 0.0072 | **1.3577 ± 0.0041** | **1.5287** | **1.3466** |
| `dlind-elu1` | 1.8352 ± 0.0020 | 1.5803 ± 0.0037 | 1.3942 ± 0.0026 | 1.5493 | 1.3865 |
| `dlind-const` | 1.8492 ± 0.0021 | 1.6236 ± 0.0039 | 1.4588 ± 0.0019 | 1.6101 | 1.4547 |

n-gramme (val entière) : 1.3402. **Écart du meilleur réseau au n-gramme : 0.0064 nat sur
la val** (1.3466 vs 1.3402) — passé de 0.48 (C7) à 0.006 ; sur le test : 1.3577.

## 2. Verdicts pré-enregistrés (tous tranchés, 5/5 seeds du même côté)

| # | Comparaison (test_best) | Différence | IC 95 % | Verdict |
|---|---|---|---|---|
| **C8-V0** | dlind-elu1 − smax @1 800 | +0.0148 | [+0.0029 ; +0.0267] | **confirmée** (réplication de C7-V1 : smax devant) |
| **C8-V1** | dlind-elu1 − smax @18 000 | +0.0364 | [+0.0314 ; +0.0415] | **infirmée** (smax reste devant) |
| **C8-V2** | dlind-elu1 − smax @5 400 | +0.0868 | [+0.0787 ; +0.0950] | **infirmée** (l'écart culmine ici) |
| **C8-V3** | dlind-const − dlind-elu1 @18 000 ≤ +0.05 | +0.0646 | [+0.0622 ; +0.0671] | **infirmée** (à 18 000 pas la version simple ne suffit plus : +0.065 > +0.05) |
| **C8-V4** | évolution de l'écart 1 800 → 18 000 | +0.0216 | [+0.0098 ; +0.0335] | **l'avantage de smax grandit** avec le budget |

Forme non monotone consignée : l'écart dlind-elu1 − smax fait **+0.015 → +0.087 → +0.036**
(1 800 → 5 400 → 18 000) : il culmine à 5 400 puis **rétrécit** à 18 000 sans se fermer.
C8-V4 (pré-enregistré sur les extrémités) dit : au total, l'avantage de smax grandit.
C8-V5 (descriptif) : meilleur réseau sur la val = smax 1.3466, n-gramme 1.3402 → le réseau
est à 0.0064 nat derrière le compteur de lettres, à 240 K paramètres.

## 3. Lecture honnête (adversariale)

1. **Résultat négatif principal : plus de budget ne fait pas gagner la delta rule.** Les
   trois IC sont sans chevauchement de 0, 5/5 seeds, trois budgets. L'avantage de smax
   même **augmente** en valeur absolue (C8-V4). Les 0.02-0.09 nat ne sont pas du bruit.
2. **La version simple ne suffit plus à 18 000 pas** (+0.065 > seuil +0.05, IC étroit
   [0.0622 ; 0.0671]) : à petit budget l'adressage par contenu ne sert à rien (C7-V2),
   à grand budget il sert un peu (C8-V3 infirmée, mais l'écart dlind-const − dlind-elu1
   rétrécit lui aussi : +0.014 → +0.043 → +0.065... non, il GRANDIT : voir chiffres).
   Correction : +0.0140 (1800) → +0.0433 (5400) → +0.0646 (18000) — l'écart entre les
   deux dlind grandit aussi : les mécanismes se spécialisent avec le budget.
3. **Le budget change tout pour la qualité absolue** : smax passe de 1.82 à **1.3577** —
   la courbe est encore descendante (pas_best = 17 400-18 000 : les runs finissent à leur
   minimum ; 5 400 aussi : pas_best = 5 400). Le n-gramme n'est plus une borne infranchissable
   (0.006 nat) alors que C3b le donnait à 0.48 au-dessus.
4. **Contre-hypothèses vérifiées** : pas de fuite (fr1m conforme, 0 copie) ; métrique
   identique entre variantes (même protocole d'éval, même test, une seule lecture) ;
   hasard des seeds exclus (5/5 du même côté à chaque budget, IC étroits) ; les poids
   « best » choisis sur la val uniquement. Le choix `test_best` (et non final) ne peut pas
   favoriser smax : test_final donne les mêmes conclusions (brut : 1.8204/1.4935/1.3580).
5. **Ce que C8 ne dit pas** : le comportement à T = 256+ (C10), sur 5× plus de données
   (fr5m_v2 — la suite indiquée par le tableau de décision, l'écart au n-gramme étant
   passé sous 0.3), et le coût relatif (C9 : dlind-const 6× plus rapide à long contexte,
   état ÷7 680 — le compromis qualité/coût reste l'argument de la delta rule).

## 4. Décision (tableau de l'étape 5)

Écart au n-gramme = 0.006-0.018 nat < 0.3 → **données d'abord** : cycle suivant sur
`data/fr5m_v2` (5× plus de texte, `--pas-eval 96`). En parallèle : C10 (T = 256) localement
(protocole commité avant). La capacité (d = 192) n'est pas déclenchée par le tableau
(l'écart n'est plus > 0.3) ; elle reste une option si fr5m_v2 stagne.
