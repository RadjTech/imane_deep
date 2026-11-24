# Rapport données — corpus français fr1m / fr5m (2026-11-24)

Mission « prendre la relève, préparer les données, valider la méthode » — étape 1.
Collecte et contrôles faits par l'agent de relève, le 2026-11-24.

## 1. Ce qui a été produit

| livrable | état |
|---|---|
| `data/fr1m/` (corpus.txt + test.txt + manifest.json) | ✔ 998 538 car. de corpus, 51 348 de test |
| `data/fr5m/` (corpus.txt + test.txt + manifest.json) | ✔ 4 998 540 car. de corpus, 251 940 de test |
| `data/test_oeuvres.txt` | ✔ 1 298 154 car. — 2 œuvres entières, auteurs absents du corpus |
| `data/SOURCES.md` | ✔ 16 œuvres : titre, auteur, mort, adresse, licence, taille, SHA-256 |
| `scripts/nettoyer_sources.py` | ✔ déterministe, sans réseau, commis |
| textes bruts + nettoyés par œuvre | hors dépôt (workspace `sources_brut/`, `sources_fr/`) |

## 2. Source unique : fr.wikisource.org

`gutenberg.org` a répondu **HTTP 503** (retry-after 21 600 s) à toutes les requêtes de
cette machine (IP de datacenter refusée) — vérifié en premier. La mission autorisait
« gutenberg.org **et** fr.wikisource.org » : tout provient donc de fr.wikisource.org
(16 pages vérifiées HTTP 200, user-agent descriptif, `maxlag=5`, pauses ; ~130 requêtes
API au total). Les URLs figurent dans `data/SOURCES.md` (toutes revérifiées HTTP 200).

## 3. Contrôles du protocole (chiffres mesurés)

**Contrôle 1 — manifestes, vocabulaire, caractères remplacés ✔**
- fr1m : vocabulaire **93** (bornes 60–160) ; remplacés par ¤ : 99 / 998 538 = **0.0099 %** (< 0.1 %).
- fr5m : vocabulaire **100** ; remplacés par ¤ : 143 / 4 998 540 = **0.0029 %**.
- `manifest.json` écrit sans erreur pour les deux (avec SHA-256 des corpus).

**Contrôle 2 — traces interdites, paragraphes ✔ (avec une réserve documentée)**
- « Project Gutenberg », « http », « @ » : **0 occurrence** partout (fr1m et fr5m, corpus et test).
- `***` : 0 dans fr1m ; **11 occurrences dans fr5m** — toutes dans *Le Rouge et le Noir*
  (Stendhal) : ce sont les **astérisques de censure originaux de l'auteur** (« le duc de
  *** », « M*** », « roi de *** »), pas du boilerplate. Les lignes-séparateurs `***`
  sont déjà supprimées par le nettoyeur. Les retirer altérerait le texte original.
- Longueur moyenne des paragraphes : fr1m corpus **231** / test **241** ; fr5m corpus
  **224** / test **242** — dans la fenêtre 150–1 500.

**Contrôle 3 — alphabet du test ⊆ alphabet du corpus ✔** pour fr1m et fr5m (manquants : aucun).

**Contrôle 4 — doublons de suites de 8 mots ✔ fr1m / ✘ fr5m (documenté, seuil non abaissé)**
- fr1m : **0.0109 %** < 0.5 % ✔.
- fr5m : **0.8603 %** > 0.5 % ✘. Examen : les 387 suites chevauchantes sont **toutes
  distinctes** (fréquence maximale 1 chacune — aucune formule ou passage dupliqué) :
  dialogues usuels (« reprit le professeur et dites moi votre nom »), titre de l'œuvre
  (« le tour du monde en quatre vingts jours »). Le dédoublonnage exact des paragraphes
  (≥ 60 car.) a retiré **3 859 doublons** avant le découpage. Conclusion : ce sont des
  récurrences naturelles du même genre littéraire, pas une fuite train→test ; à 4,2 Mo
  de train, toute séquence courante de 8 mots français a de fortes chances d'exister.
  **Le seuil n'a pas été abaissé** : le contrôle est consigné en échec formel sur fr5m,
  avec ces éléments pour jugement.

**Contrôle 5 — essai de fumée ✔** `smax,smax` sur fr1m, 10 pas, seed 1 : perte
4.5525 → 4.2447 (baisse sans erreur).

**Contrôle 6 — cinq extraits de 300 caractères ✔** (fr1m, corpus) :

1. « Pendant qu'ils devisaient de ces choses extrêmement délicates, des champs, des lacs, des forêts, des montagnes se déroulaient devant eux et, toujours, à quelque tournant, à travers le frais tamis de cette perpétuelle ondée qui suivait le héros dans ses excursions, la Jungfrau dressait sa cime blanch… »
2. « Mais un certain nombre de voyageurs avaient été immédiatement séduits par la proposition. Elle plaisait particulièrement au colonel Proctor. Ce cerveau brûlé trouvait la chose très-faisable. Il rappela même que des ingénieurs avaient eu l'idée de passer des rivières « sans pont » avec des trains rig… »
3. « — Voici ce que c'est, dit la petite Fadette en soulevant son panier et en le plaçant entre les jambes du père Barbeau. Ma défunte grand'mère avait gagné dans sa vie, à donner des consultations et à vendre des remèdes, plus d'argent qu'on ne pensait ; comme elle ne dépensait quasi rien et ne plaçait… »
4. « — De cette façon, disait-il, vous créerez une société énergique, ayant appris depuis longtemps à connaître et à aimer le pays, sachant sa langue et au courant de toutes ces graves questions locales auxquelles se heurtent infailliblement les nouveaux venus. »
5. « À six heures du soir, le Mongolia battait des branches de son hélice les eaux de la rade d'Aden et courait bientôt sur la mer des Indes. Il lui était accordé cent soixante-huit heures pour accomplir la traversée entre Aden et Bombay. Du reste, cette mer indienne lui fut favorable. Le vent tenait dan… »

## 4. Répartition par auteur (mesurée sur les corpus finis)

| corpus | part maximale | détail |
|---|---|---|
| fr1m (1,001,759 car.) | Dumas **13.4 %** | Zola 12.3, Flaubert 12.2, Verne 11.7, Hugo 11.6, Balzac 9.8, Sand 9.5, Maupassant 8.4, Stendhal 7.1, Daudet 4.0 |
| fr5m (5,000,834 car.) | Dumas **13.9 %** | Zola 13.1, Flaubert 12.8, Hugo 12.4, Verne 10.4, Balzac 9.6, Stendhal 8.6, Maupassant 8.4, Sand 7.3, Daudet 3.6 |

10 auteurs dans le corpus (≥ 8 exigés), 14 œuvres de prose (≥ 10 exigées), aucune part
> 20 % ✔. Plafond appliqué au nettoyage : 1 000 000 de caractères nettoyés par auteur,
répartis proportionnellement entre ses œuvres (aucune œuvre réduite à zéro) ; les
prélèvements sont consignés par œuvre dans `data/SOURCES.md`.

## 5. Décisions et limites à connaître

- **gutenberg.org inaccessible depuis ce bac à sable** (503, IP datacenter) : source
  unique fr.wikisource.org, comme autorisé par la mission. Si tu préfères Gutenberg,
  il faut fournir les fichiers ou une autre machine.
- Les œuvres très longues (> 1 Mo nettoyé) sont prélevées proportionnellement :
  documenté œuvre par œuvre dans SOURCES.md (l'alternative — garder les œuvres
  entières — faisait dépasser la règle des 20 % par auteur).
- `test_oeuvres.txt` : œuvres entières d'auteurs hors corpus (Huysmans, About),
  nettoyées par le même script ; jamais utilisées pour choisir quoi que ce soit.
- La poésie, le théâtre et les tableaux n'ont pas été téléchargés (sources choisies
  dans les romans) ; les titres de chapitres, numéros de page, illustrations, notes
  sont retirés par `nettoyer_sources.py` (règles dans son en-tête).
- Machine de collecte : 2 cœurs, 2 Go de RAM (~1.5 Go libre), Python 3.13.18, NumPy 2.3.5.
