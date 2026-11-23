# Protocole de recherche — la discipline des cycles

> Règle fondatrice : **une expérience non traçable n'a jamais eu lieu.** Chaque cycle passe par les 9 étapes ci-dessous, dans l'ordre, sans les sauter — y compris (surtout) quand le résultat contredit l'hypothèse.

## Méthode : agentic experiments + get-research-done

**Agentic experiments** : les cycles sont exécutés par un agent qui enchaîne automatiquement garde-fous, entraînements, mesures et journalisation. L'agent n'a pas le droit de :
- entraîner sans garde-fou gradient validé (étape 4) ;
- modifier le protocole entre deux configurations comparées ;
- déclarer un « résultat » sans chiffre reproductible (seed, commande exacte) ;
- tirer une conclusion générale d'une seule configuration sans répétition de seed.

**Get-research-done** : un cycle est *terminé* seulement si la question de départ reçoit une réponse exploitable (poursuivre / pivoter / arrêter) consignée dans `RESEARCH_LOG.md`. Pas de cycle ouvert sans décision.

## Les 9 étapes d'un cycle

1. **Question** — une question précise, mesurable, écrite en tête de l'entrée du journal.
2. **Hypothèse** — falsifiable, avec un **critère de décision numérique** annoncé à l'avance (« si X alors Y de Z % »). Ajoutée à `HYPOTHESES.md` avec son statut `à tester`.
3. **Protocole** — commande exacte, seeds, baselines, ce qui varie et ce qui est figé. Le protocole est commité **avant** l'expérience.
4. **Garde-fou** — `--check-grads` doit passer (écart relatif < 1e-4) sur le code modifié. Sans ça, pas d'étape 5.
5. **Expérience** — exécution du protocole. Les logs bruts sont conservés dans `results/`.
6. **Analyse** — chiffres en table, comparaison aux baselines, lecture honnête (un résultat négatif est un résultat).
7. **Journal** — entrée complète dans `RESEARCH_LOG.md` : question, hypothèse, protocole, résultats, analyse, décision.
8. **Décision** — poursuivre (H+1) / pivoter / arrêter ; mise à jour de `HYPOTHESES.md`.
9. **Traçabilité** — commit + push (message : `C<k>: <question> → <décision>`), la CI rejoue les garde-fous.

## Les 5 règles de mesure

1. **Ablation propre** : une seule chose varie entre configurations ; tout le reste (paramètres, données, optimiseur, steps, seed) est identique et écrit dans le journal.
2. **Seeds** : toute conclusion de tendance doit être confirmée sur ≥ 3 seeds avant d'être consignée comme « établie » (sinon : « indication »).
3. **Métriques CPU-first** (définies dans `METRICS_CPU.md`) : qualité **et** coût mémoire/vitesse, jamais la seule perte.
4. **Baselines figées** : les CSV de baseline sont commités une fois et ne bougent plus (sinon plus de référence).
5. **Honnêteté** : les limites et les échecs sont consignés avec le même soin que les succès — c'est ce qui rend le dépôt scientifique.

## Correspondance des types de couches

| code | mécanisme | coût séquentiel | mémoire d'inférence |
|---|---|---|---|
| `smax` | attention softmax causale (module 4 du cours) | O(T²) | KV-cache : 768 octets/token/couche qui croît avec T |
| `lin`  | attention linéaire à état φ(k)⊗v (module 6) | O(T) | état constant : 2·d²·4 octets par couche (≈ 73 Ko ici), indépendant de T |
