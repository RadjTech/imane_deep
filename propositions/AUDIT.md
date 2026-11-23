# Audit indépendant — imane_deep (commit `c58eb03`, numéros de ligne de ce commit)

HEAD est passé à `2a976e8` (C4, branche `c4-releve`) à 17:42 pendant la rédaction : lu seulement (§7), mes mesures portent sur `c58eb03` et C4 ne touche aucune fonction testée. Non re-démontré : φ ≥ 1, KN ≈ 1,65, reproduction ±0,002. LOG = RESEARCH_LOG.md, HYP = HYPOTHESES.md, MET = METRICS_CPU.md.

## 1. Constats par gravité (mesurés)

| # | Constat et preuve | Où | Invalide |
|---|---|---|---|
| **G1** | **C3 fait varier 3 choses** : mise à jour de S, clés L2 (donc Z aussi), β appris. À l'init (corpus réel), poids effectifs de dlind par retard 1..5 = 0,34/0,18/0,095/0,05/0,027 (lin, smax : 0,013 plat) ; cos(k̂) moyen 0,947, min 0,879 ; RMS(y_t) ∝ t^−1,01 (lin −0,26) ; ‖S‖ 0,5→1,7 mais ‖Z‖ 1→93. dlind ≈ EMA(v)/t sur 2-3 tokens | LOG:129, HYP:14, hybrid_numpy.py:142 contre :228-231 | l'attribution causale de H2, pas le chiffre 1,8976 |
| **G2** | **H1 « confirmée » viole les règles annoncées** : H1b infirmée ⇒ H1 « retombe à indication » (non appliqué) ; H0 : 2×0,058 = 0,116 > écart lin−hybride 0,083, et analyse.json dit `interpretable:false`. Rattrapage par seed **111 / 30 / 98 %**, 63 % sans s1337, p = 0,094. « Résidu 0,103 » faux : hybride−smax = 0,0204 | HYP:34 contre :10, :29 ; LOG:103,110 ; analyse_c2.py:84 (« conclusion **fine** ») | H1 = indication |
| **G3** | **C1 non pré-enregistré** : `440e03e` (12:05:46) affiche « ✅ terminé », « H1 confirmée : 2,1265 (pas 500) » alors que son log s'arrête au pas 300 (pas 500 réel : 2,1972). 2,1265 / 2,5496 / 2,1214 absents de toute donnée. Run démarré ≤ 12:04:06, avant le commit. Critère 2,175→2,179 dans le commit du résultat (`498cc0a`), sans effet | LOG:12 garde 2,1214 ; LOG:15,36 contre HYP:9 | H1 comme prédiction |
| **G4** | **H4 = HARKing, mécanisme démenti** : critère et résultat dans `5f6ecc9` ; val@1800 hors d'analyse_c2.py, qui interdit toute autre analyse. train@1800 : dlind 1,018 ≈ lin 0,992. lin,lin (2 couches d'état) val@1800 2,786 > hybride 2,552 ; pente hybride +0,386 **omise**. H4b (2,914) déjà passé par lin (2,786). L'effet 0,56 nat est réel (s2024/s777 : 0,54/0,47) | analyse_c2.py:6-7 ; LOG:163 ; HYP:20,40 | « l'état borne la mémorisation » |
| **G5** | **Profil `lin,lin` écrasé** en `3ec42cc` par un ckpt dlind `/tmp/smoke_dlind` (961 952 o) : 3 128/3 092/**1 552** tok/s, pas 2 494/2 652/2 693. « ±20 % » démenti par les JSON : dlind @4096 1 616 puis 2 583. `8a36b3c` « équivalence validée » : avec types dlind le garde-fou échoue (0,23 à 3,3 ; seuil 2e-3) | LOG:93,98,106,156 ; MET:44,50,54 | « plat », « ±20 % » |
| **I1** | smax du profileur sans BLAS : 4,39× annoncé → ≈ 2,5× (§5) | profiler_inf.py:95,99 | l'ampleur de l'avantage |
| **I2** | Mémoire : état réel 9 600 o/couche, pas 73 728 (×7,68). M@96 = 147 456 (smax, juste) / **83 328** (hybride) / **19 200** (lin), pas 147 Ko ×3 | RESEARCH_PROTOCOL.md:40 ; MET:16-17 ; LOG:29-31,38 | tableau M@T |
| **I3** | « Variance la plus faible » : lin 0,0117 < dlind 0,0131. n=3 ⇒ IC95 de σ [0,007 ; 0,082] ; hybride vs smax p = 0,09 | HYP:19 ; LOG:103,164,171 | « robuste / instable au seed » |
| **I4** | H3 : énoncé ⇒ critère, réciproque fausse (Δin 0,5 ; Δptq 1,2 ⇒ « confirmée » mais 0,5 > 0,15) | HYP:23-24 | H3 |
| **I5** | Suivre le README écrase des données versionnées : localement 2,1132 au lieu de 2,1112 ; analyse_c2 donnerait 2,1658 ; c1_baseline_lin tronqué au pas 1300 | README.md:41-43 | règle 4 |
| **m** | ±0,047→0,0487 (MET:39) ; 2 610→2 622 (LOG:95, MET:52) ; 942 Ko→939 Kio (MET:18) ; 6,3 contre 6,1 Mo (LOG:29/96) ; ~9→10,9 min (LOG:33) ; « dépasse de 340 % »→240 % (LOG:162) ; seuil 1e-3 contre 2e-3 (LOG:65, profiler:159) ; « plus lent que les baselines » faux pour lin (3 950 contre 3 628-3 828 tok/s, LOG:166) ; β→0 n'est pas lin (β=1e-6 : RMS y = 6e-8, LOG:165) ; dernier token du prompt rejoué (profiler:171,205) ; H0 « en cours » (HYP:30) | | |

## 2. Conformité aux références (impact probable)

Mesuré : AdamW, LayerNorm, clip, GELU-tanh = torch (≤ 1e-15) ; schedule et init conformes. Divergences :
- **État exclusif** pour lin/dlind (y_0 = 0 mesuré) alors que smax inclut le courant, comme les références (*mémoire*) : README.md:46 « ablation propre » inexact dès C1. Impact faible à moyen.
- **Z cumulé gardé dans dlind** : S borné, Z ∝ t ⇒ sortie en 1/t, 33× plus faible que lin à t=95. Impact fort ; Schlag 2021 déconseille ce dénominateur (*mémoire*).
- **β par tête, constant**, au lieu de β_t = σ(W x_t) (*mémoire*) ; γ de 0,16-0,41 pour un déplacement Adam maximal de 0,99 : proche de l'init, pas un « choix ».
- **Normalisation** : L2 vient de Yang 2024, pas de Schlag (somme de k et q, *mémoire*) ; absente de lin. φ ≥ 1 ⇒ cos ≥ 0,88 : aucune adresse distincte, l'écriture « remplaçante » est un oubli exponentiel.
- Sans dropout ni réglage par type ; val et train partagent le même RNG (hybrid_numpy.py:449,466) : `log_every` change les batchs. 1 801 mises à jour ; le dernier caractère de val n'est jamais prédit.

## 3. Pré-enregistrement

| H | Critère | Résultat | Modifié / données d'origine |
|---|---|---|---|
| H0 | 440e03e ; règle fda6633 | 5f6ecc9 | [0,02 ; 0,04] prédit : faux 3/3 (0,018/0,012/0,058) ; règle affaiblie puis non appliquée |
| H1 | 440e03e, **avec** « confirmée » | 498cc0a | 2,175→2,179 dans le commit du résultat ; puis seed → moyenne incluant s1337 |
| H1b | fda6633 | 5f6ecc9 | non ; s1337 de découverte = 1/3 des données ; indistinguable (+0,020, IC [−0,09 ; +0,13]) |
| H2 | 440e03e (2,175) → fbdad4b (2,1964) | c58eb03 | avant les runs, sans effet ; pas de HARKing |
| H3 | 440e03e | — | non équivalents (I4) |
| H4 | 5f6ecc9 | 5f6ecc9 | HARKing (idée de 498cc0a) |
| H4b | fbdad4b | c58eb03 | non ; non discriminant (lin passe) |

Horodatage cohérent : C2 démarre entre f3342cf (12:25:06) et 12:25:49 ; C3 entre fbdad4b (14:15:33) et 14:15:51. Smoke dlind antérieur à 14:22, non daté.

## 4. Statistique

Bruit d'une évaluation (160 fenêtres) : 0,010 à 2 nats (proxy n-gramme), 0,024-0,038 (ckpt locaux, pas 1300-1800) ; l'écart type inter-seeds publié (0,012-0,018) est donc en partie ce bruit. Biais du minimum ≈ −0,012 à −0,024, différentiel dlind−smax −0,006. Erreur type due au texte (blocs de 50) : écart apparié 0,02-0,04 (n-grammes), 0,07-0,105 (modèles sur-appris, borne haute) ⇒ dlind−smax 2,4 à 10 σ ; lin−smax 1 à 5 σ ; hybride−smax ≤ 1 σ. Trois seeds = **un seul** texte. Fuite : phrase de 67 car. (fin de val) dans le train ; 8 % de la val en sous-chaînes ≥ 20. Hors git ⇒ invérifiable : β/γ (LOG:165), équivalences 3,7e-05 / 9,2e-06 / 0,589 / 7,15e-07, tout test sur un autre texte ; c1_synthese.json sans script. Les ckpt sont ceux du pas 1800, pas du val_min.

## 5. Profileur (min de blocs, machine chargée par 3 entraînements, 35 s)

tok/s à T = 96/256/1024/4096 : smax einsum 784/695/495/208 ; smax BLAS 967/848/649/363 ; dlind 678/676/710/705. dlind/smax : 0,86/0,97/1,43/**3,39** → BLAS 0,70/0,80/1,09/**1,94** (gain ×1,74 @4096) ; sur les 588 du dépôt ≈ 1 020, d'où 4,39× → ≈ 2,5×. Le mur O(T) reste réel. Dans la distribution, le JSON du dépôt donne dlind/smax = **0,77/0,62/0,72** (T0 = 16/48/96) : prévu (LOG:66), jamais publié. 1 789 < 2 583 : compatible avec ±20 % seulement si V ∈ [2152 ; 2236] ; les 6 mesures dlind (1 670-2 583) l'excluent.

## 6. Vérifié CORRECT

Gradients exacts, tous paramètres, autograd torch float64 (7 combinaisons, init et poids bruités, d=16 et 96) : ≤ 1e-15 (smax, lin), ≤ 3,4e-10 (dlind ; ≤ 9e-16 avec le 1e-8 de norm_keys_bw). **6,86e-05 = arrondi** : `b1.qkv.b[5]`, g = −5,87e-07, erreur 4,0e-11 < eps_mach·L/eps = 9,7e-11, 5,6e-07 à eps=1e-3. Garde-fou fragile : ≈ 25 % (lin+smax) et 9 % (dlind) de fausses alarmes sur un backward exact. À eps ≥ 3e-3 la différence finie **voit** le saut de φ (×7 à ×25). Tous les val_min, moyennes, écarts types, val@1800, 80 % / 340 % et V_gen (sauf G5) concordent avec CSV/JSON ; verdicts H2/H4b justes et pré-enregistrés ; baselines inchangées ; float32/float64 ≤ 1e-6.

## 7. C4 avant les runs

Δ_A = val_min (19 évals aléatoires) − 1,8976 ; Δ_B = éval fixe sur le ckpt du **pas 1800** (hybrid_numpy.py:617 écrase à chaque éval ; val ≈ 2,03) : deux estimateurs, deux points de fonctionnement. Ckpt C3 absents du dépôt. `evaluer_fixe` couvre 1 056/1 117 cibles. std ddof=0 (analyse_c4.py:36) contre ddof=1. Mémoire ×7,26 calculée, non mesurée (README : 16-32×). `pipefail` absent ; STE non testé.

## 8. Supposé

Attributions Schlag/Yang et défauts GPT-2 : de mémoire. Vos contrôles `exp_phi` (const : 1,925@600 contre 1,929 pour dlind, non conduits par moi) sont compatibles avec G1. Écart aux consignes : cinq copies `.md` écrites dans `/tmp` puis supprimées.
