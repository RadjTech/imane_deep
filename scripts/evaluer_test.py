#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Évaluation de TEST d'un run C7 — fenêtres non chevauchantes, poids finaux.

Relit checkpoint.npz + resume.json du run, réinstalle la variante
(c3b_controles.installer), reconstruit le vocabulaire depuis corpus.txt,
découpe le texte évalué en fenêtres NON chevauchantes de 96 caractères
(chaque caractère prédit exactement une fois, aucun tirage aléatoire),
remplace les caractères dont le compte dans le train est < --min-compte
(20 par défaut, comme preparer_corpus.py) par ¤ et les compte.

Contrôle obligatoire (--controle) : sur la val du corpus 11 Ko, avec
min-compte 0, doit reproduire evaluer_fixe de hybrid_numpy.py à < 1e-9.
"""
import argparse
import json
import os
import resource
import sys

RACINE = os.environ.get("IMANE_RACINE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "experiments"))
sys.path.insert(0, os.path.join(RACINE, "scripts"))
import numpy as np  # noqa: E402
import hybrid_numpy as hn  # noqa: E402
import c3b_controles as cc  # noqa: E402

INCONNU = "¤"


def cartes_vers_ids(texte, stoi, gardes, compte_inconnu):
    """Cartes remplacées par ¤ (comptées), puis ids. Retourne (ids, n_inconnus)."""
    n = 0
    ids = []
    for c in texte:
        if c not in gardes:
            n += 1
            c = INCONNU
        ids.append(stoi[c])
    return np.array(ids, dtype=np.int32), n


def pertes_fenetres(P, cfg, ids, T):
    """Même protocole que hn.evaluer_fixe : fenêtres non chevauchantes de T (B=1)
    plus une dernière fenêtre plus courte, chaque caractère prédit une fois,
    moyenne pondérée par caractère."""
    somme, n, i, L = 0.0, 0, 0, len(ids)
    while i < L - 1:
        r = min(T, L - 1 - i)
        x = ids[i:i + r][None, :]
        y = ids[i + 1:i + r + 1][None, :]
        somme += float(hn.forward(P, cfg, x, y)[1]) * r
        n += r
        i += r
    return somme / n, n


def main():
    ap = argparse.ArgumentParser(description="Évaluer le test d'un run C7")
    ap.add_argument("--run", required=True, help="dossier du run (checkpoint.npz + resume.json)")
    ap.add_argument("--data", required=True, help="corpus.txt (vocabulaire + découpage train)")
    ap.add_argument("--test", default=None, help="fichier test (défaut : <dossier corpus>/test.txt)")
    ap.add_argument("--test-oeuvres", default=None, help="data/test_oeuvres.txt si présent")
    ap.add_argument("--min-compte", type=int, default=20,
                    help="compte minimal dans le train pour garder un caractère (0 : tout garder)")
    ap.add_argument("--controle", action="store_true",
                    help="vérifie l'équivalence avec evaluer_fixe sur la val (corpus 11 Ko, min-compte 0)")
    ap.add_argument("--out", default="test.json")
    a = ap.parse_args()

    resume = json.load(open(os.path.join(a.run, "resume.json")))
    cc.installer(resume["phi"], resume["cles_normalisees"], resume["delta"])
    try:
        texte, ids, stoi, itos = hn.charger_donnees(a.data)
        n_train = int(0.9 * len(ids))
        train = "".join(itos[i] for i in ids[:n_train])
        from collections import Counter
        gardes = {c for c, k in Counter(train).items() if k >= a.min_compte} \
            if a.min_compte > 0 else set(itos)
        cfg = {"vocab_size": len(itos), "d_model": 96, "n_head": 4,
               "types": resume["types"], "block_size": 96}
        P = dict(np.load(os.path.join(a.run, "checkpoint.npz")))
        T = cfg["block_size"]
        res = {"run": a.run, "variante": resume.get("variante"), "seed": resume.get("seed"),
               "min_compte": a.min_compte, "fenetre": T}

        if a.controle:
            # même découpage que hybrid_numpy : val = derniers 10 %
            data_val = ids[n_train:]
            perte_moi, n_car = pertes_fenetres(P, cfg, data_val, T)
            perte_ref = hn.evaluer_fixe(P, cfg, data_val, T)
            ecart = abs(perte_moi - perte_ref)
            res.update({"val_moi": perte_moi, "val_ref": perte_ref,
                        "ecart": ecart, "caracteres_predits": n_car})
            ok = ecart < 1e-9
            res["controle"] = "✔ <1e-9" if ok else "✘ ÉCART TROP GRAND"
            print(json.dumps(res, ensure_ascii=False))
            sys.exit(0 if ok else 1)

        chemin_test = a.test or os.path.join(os.path.dirname(a.data), "test.txt")
        t_test = open(chemin_test, encoding="utf-8").read()
        ids_test, inc_test = cartes_vers_ids(t_test, stoi, gardes, 0)
        perte_test, nf_test = pertes_fenetres(P, cfg, ids_test, T)
        res.update({"test": perte_test, "fenetres_test": nf_test,
                    "inconnus_test": inc_test, "car_test": len(t_test)})
        if a.test_oeuvres and os.path.exists(a.test_oeuvres):
            t_oe = open(a.test_oeuvres, encoding="utf-8").read()
            ids_oe, inc_oe = cartes_vers_ids(t_oe, stoi, gardes, 0)
            perte_oe, nf_oe = pertes_fenetres(P, cfg, ids_oe, T)
            res.update({"test_oeuvres": perte_oe, "fenetres_oeuvres": nf_oe,
                        "inconnus_oeuvres": inc_oe, "car_oeuvres": len(t_oe)})
        res["rss_mo_eval"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        chemin = os.path.join(a.run, a.out)
        json.dump(res, open(chemin, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("RÉSUMÉ :", json.dumps(res, ensure_ascii=False))
    finally:
        cc.desinstaller()


if __name__ == "__main__":
    main()
