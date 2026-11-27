# Bilan final de la mission « réparer, mesurer, conclure » (2026-11-26)

> Couvre : réparations (étape 1), C8 (45 runs GitHub), C9 (coût), C10 (contexte long,
> 15 runs dont amendement), C11 (données, 9 runs GitHub). Plafond 72 h : ~19 h de calcul
> local utilisées (hors GitHub Actions). Tous les chiffres sont mesurés, vérifiés par un
> second script indépendant et commités avec leur protocole daté. Détails :
> `propositions/RESULTATS_C8.md`, `RESULTATS_C9.md`, `RESULTATS_C10.md`, `RESULTATS_C11.md`,
> journal complet dans `RESEARCH_LOG.md` et `AVANCEMENT.md`.

## 1. Notre méthode (delta rule) est-elle utile ?

**En qualité pure : non.** L'attention classique (`smax`) gagne partout où le critère est
tranché : à 1 800 pas (C7-C8-V0), à 5 400 et 18 000 pas (C8), sur plus de données (C11).
C'est un résultat négatif, mesuré proprement, et il contredit ce que suggérait le tout
petit corpus de C3b : l'avantage observé là-bas était un effet de bachotage, pas un
avantage de mécanisme.

**En coût : oui, nettement.** À long contexte, la version simple (`dlind-const`) génère
**6,1× plus vite** que `smax` (5 513 vs 899 tokens/s à 4 096 tokens) avec un état de
**0,8 Ko contre 6,1 Mo** (facteur 7 680), et `dlind` standard est 4,2× plus vite avec 19 Ko.
Et à contexte long (T = 256), le désavantage de qualité **disparaît** : différence
−0.0029, IC [−0.0166 ; +0.0107] (6 seeds), contre +0.0868 [+0.0787 ; +0.0950] à T = 96
(IC disjoints — C10). La delta rule rattrape exactement là où elle est la moins chère.

## 2. Dans quel cas l'utiliser ?

| Situation | Choix | Pourquoi |
|---|---|---|
| Qualité maximale, contexte ≤ 96-256, mémoire abondante | `smax` | devant à tous les budgets testés (écart +0.02 à +0.09 nat) |
| Génération de **longs** textes, mémoire contrainte, débit | `dlind-const` | 6× plus vite à T = 4096, état 0.8 Ko, et plus de désavantage de qualité détecté à T = 256 |
| Compromis intermédiaire | `dlind-elu1` | +2.7 % de perte seulement à 18 000 pas, état 19 Ko, vitesse plate à tout contexte |

## 3. À quel prix ?

Sur fr1m, à 18 000 pas (le meilleur régime mesuré) : `dlind-elu1` coûte **+2.7 % de perte**
par rapport à `smax`, `dlind-const` **+7.5 %**. En échange : ×4-6 de vitesse en génération
longue et ÷100-7 680 de mémoire d'état. La version simple ne « suffit » plus strictement
aux grands budgets (bornes d'IC +0.0525 à T=256 et +0.0671 à 18 000 pas, seuil +0.05) :
l'adressage par contenu sert un peu quand l'entraînement est long ou riche.

## 4. Entre combien de lettres le modèle hésite (exp de la perte — plus petit = mieux)

| Méthode | Perte | Hésite entre… |
|---|---|---|
| **Compteur de lettres (n-gramme)** | 1.3402 | **3,8** |
| **Attention classique (`smax`)** — 18 000 pas | 1.3577 | **3,9** |
| **`dlind-elu1`** — 18 000 pas | 1.3942 | **4,0** |
| **`dlind-const`** — 18 000 pas | 1.4588 | **4,3** |
| *rappel C7 (1 800 pas) : smax* | *1.8223* | *6,2* |
| *fr5m_v2 à 18 000 pas (autre corpus, incomparable) : smax / compteur* | *1.3008 / 1.2220* | **3,67 / 3,39** |

Le chemin parcouru : 6,2 → 3,9 lettres d'hésitation (attention) sur fr1m en passant de
1 800 à 18 000 pas ; à 18 000 pas le meilleur réseau est à **0.006 nat du compteur** sur
la validation (contre 0.48 au départ). Le compteur reste devant à budget fixe — et il
profite des données plus vite que le réseau sous-budget (C11 : 0.25 d'écart à b5400 sur
fr5m_v2, car le réseau ne voit que 1,8 époque) : **le budget doit croître avec les
données**.

## 5. Qu'est-ce qui reste inconnu ?

- **fr5m_v2 + 18 000 pas : MESURÉ (C11-bis)** — le budget referme le trou avec le
  compteur (0.252 → 0.067 nat, pente ×4 cohérente avec fr1m). Suite mesurable : 60 000
  pas (~1,7 h/run sur GitHub) — non lancé, protocole à écrire avant.
- **T = 256 (ou plus) à budget long** : C10 a montré le rapprochement à 5 400 pas ;
  à 18 000, rien n'est mesuré.
- **Capacité d = 192** : non déclenchée par le tableau de décision, non mesurée.
- **Kernels entiers / basse précision** (l'argument historique de la delta rule) : hors
  périmètre NumPy, jamais mesuré ici — l'avantage coût repose sur les formes, pas sur
  des kernels spéciaux.
- **C6 (budget)** : réservé, non touché.
- 3 seeds pour C10-V2 et C11 (minima du protocole) ; C10-V1 a 6 seeds.

## 6. Ce que je referais autrement

1. **Nettoyer les textes de test avec le même pipeline que le corpus dès le premier jour**
   (défaut 1 : 13 294 ¤ dans test_oeuvres, chiffres C7 invalides puis réparés).
2. **Dédoublonnage par fenêtre glissante dès le premier corpus** (défaut 2 : fr5m v1
   contenait des répliques répétées train↔val/test, à reconstruire en fr5m_v2).
3. **Comparer les courbes à pas égaux dès le départ** (le « 1 800 pas trop court » a été
   constaté après coup par le relecteur, pas par mes analyses).
4. **Dimensionner le budget avec les données** (C11 : plus de données à budget fixe
   n'aide pas le réseau, seulement le compteur).
5. **Lire les gardes-fous comme des tests d'intégration dès l'écriture** (deux bugs de
   mes scripts de vérification — CDF du quantile, verrouillage d'index — ont été attrapés
   par mes propres écarts, mais après coup).

## 7. État du dépôt

Tout est sur `main`, poussé : protocoles datés avant les runs, 108 runs de cycles C7-C11
documentés, réparations des deux défauts, vérifications indépendantes toutes IDENTIQUES,
aucune branche, C6 intact, corpus protégés (`.gitattributes`, SHA-256 vérifiés).
Prochaine action suggérée (à déclencher sur GitHub par l'utilisateur) : workflow C11-bis,
`fr5m_v2` + 18 000 pas.
