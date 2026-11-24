# Sources du corpus français (fr1m / fr5m / test_oeuvres)

**Date de collecte** : 2026-11-24. **Source** : fr.wikisource.org uniquement —
gutenberg.org a refusé les requêtes de cette machine (HTTP 503, IP de datacenter) ;
la mission autorisait gutenberg.org ou fr.wikisource.org.

**Licence** : œuvres du domaine public (auteurs morts avant 1910, aucune traduction) ;
la transcription Wikisource est publiée sous CC BY-SA 4.0 ou domaine public selon la page.
Textes bruts et nettoyés : hors dépôt ; seuls les empreintes SHA-256 ci-dessous et
les corpus préparés (`data/fr1m`, `data/fr5m`) sont dans le dépôt.

Nettoyage : `scripts/nettoyer_sources.py` (déterministe, sans réseau) ; découpage :
`scripts/preparer_corpus.py` (train/val/test par documents, dédoublonnage des
paragraphes exacts, caractères rares → ¤).

| fichier nettoyé | œuvre | auteur | mort | adresse | taille (car.) | SHA-256 | note |
|---|---|---|---|---|---|---|---|
| `Balzac_Le_P_re_Goriot.txt` | Le Père Goriot | Honoré de Balzac | 1850 | https://fr.wikisource.org/wiki/Le_Père_Goriot_(1910) | 531,263 | `df9a3f2389290d6d…` |  |
| `Balzac_Eug_nie_Grandet.txt` | Eugénie Grandet | Honoré de Balzac | 1850 | https://fr.wikisource.org/wiki/Eugénie_Grandet | 377,096 | `8f8a854b1b311df0…` |  |
| `Zola_Germinal.txt` | Germinal | Émile Zola | 1902 | https://fr.wikisource.org/wiki/Germinal | 1,000,530 | `447ea8765507fec1…` | prélèvement proportionné (plafond 1 Mo par auteur) |
| `Verne_Le_Tour_du_monde_en_quatre_vingts_jours.txt` | Le Tour du monde en quatre-vingts jours | Jules Verne | 1905 | https://fr.wikisource.org/wiki/Le_Tour_du_monde_en_quatre-vingts_jours | 558,368 | `76cb8b99f3128d01…` | prélèvement proportionné |
| `Verne_De_la_Terre_la_Lune.txt` | De la Terre à la Lune | Jules Verne | 1905 | https://fr.wikisource.org/wiki/De_la_Terre_à_la_Lune | 441,946 | `b501b34ed91ff878…` | prélèvement proportionné |
| `Dumas_La_Reine_Margot.txt` | La Reine Margot | Alexandre Dumas | 1870 | https://fr.wikisource.org/wiki/La_Reine_Margot_(Dumas) | 1,000,016 | `f054ef6a8ac66577…` | prélèvement proportionné ; pages « Texte entier » tomes 1-2 |
| `Flaubert_Madame_Bovary.txt` | Madame Bovary | Gustave Flaubert | 1880 | https://fr.wikisource.org/wiki/Madame_Bovary | 1,000,348 | `d2436cb94e5a6e78…` | prélèvement proportionné |
| `Maupassant_Bel_Ami.txt` | Bel-Ami | Guy de Maupassant | 1893 | https://fr.wikisource.org/wiki/Bel-Ami | 1,000,090 | `2c3b22c08c2f708a…` | prélèvement proportionné ; édition Conard 1910 |
| `Hugo_Notre_Dame_de_Paris.txt` | Notre-Dame de Paris | Victor Hugo | 1885 | https://fr.wikisource.org/wiki/Notre-Dame_de_Paris | 622,409 | `50be8959dbc5920c…` | prélèvement proportionné |
| `Hugo_Quatre_vingt_treize.txt` | Quatre-vingt-treize | Victor Hugo | 1885 | https://fr.wikisource.org/wiki/Quatre-vingt-treize | 378,361 | `8be9fbd9f659693a…` | prélèvement proportionné |
| `Sand_La_Mare_au_diable.txt` | La Mare au diable | George Sand | 1876 | https://fr.wikisource.org/wiki/La_Mare_au_diable | 385,302 | `ae290c5ce7fbf4ed…` |  |
| `Sand_La_Petite_Fadette.txt` | La Petite Fadette | George Sand | 1876 | https://fr.wikisource.org/wiki/La_Petite_Fadette | 615,080 | `a03999dadbdf3498…` | prélèvement proportionné |
| `Stendhal_Le_Rouge_et_le_Noir.txt` | Le Rouge et le Noir | Stendhal (Henri Beyle) | 1842 | https://fr.wikisource.org/wiki/Le_Rouge_et_le_Noir | 569,370 | `2ebd86841345e410…` |  |
| `Daudet_Tartarin_sur_les_Alpes.txt` | Tartarin sur les Alpes | Alphonse Daudet | 1897 | https://fr.wikisource.org/wiki/Tartarin_sur_les_Alpes | 285,717 | `8f83a5d022ace260…` |  |
| `Huysmans_rebours.txt` | À rebours | Joris-Karl Huysmans | 1907 | https://fr.wikisource.org/wiki/À_rebours | 858,498 | `3bac58e1681ea883…` | JEU DE TEST (auteur absent du corpus) |
| `About_Le_Roi_des_montagnes.txt` | Le Roi des montagnes | Edmond About | 1885 | https://fr.wikisource.org/wiki/Le_Roi_des_montagnes | 375,924 | `8591106452505c6d…` | JEU DE TEST (auteur absent du corpus) |

## Contraintes respectées

- 10 auteurs différents dans le corpus (Balzac, Zola, Verne, Dumas, Flaubert,
  Maupassant, Hugo, Sand, Stendhal, Daudet) ; 14 œuvres de prose ; le jeu de test
  (`data/test_oeuvres.txt`) provient de 2 auteurs absents du corpus (Huysmans, About).
- Plafond par auteur de 1 000 000 de caractères nettoyés (répartition proportionnelle
  entre les œuvres d'un même auteur) : part maximale mesurée dans les corpus finis =
  13,9 % (Dumas, fr5m) < 20 %.
