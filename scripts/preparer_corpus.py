#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prépare un corpus caractère avec train / val / TEST, sans toucher à hybrid_numpy.py.

hybrid_numpy.py coupe un fichier unique à 90/10. Ce script produit donc :
  corpus.txt   = train puis val, coupés exactement à 90/10 (utilisable tel quel par --data)
  test.txt     = jamais vu, à n'évaluer qu'UNE fois, en fin de cycle
  manifest.json = sources, licence, empreintes SHA-256, vocabulaire, statistiques

Garanties : découpage par DOCUMENT (pas au milieu d'un texte), paragraphes en double
retirés avant découpage (pas de fuite train → val/test), caractères rares remplacés
(aucun caractère propre à la val), tirage reproductible par --seed.

Usage : python3 scripts/preparer_corpus.py --sources a.txt b.txt --nom wiki5m \
            --licence "CC BY-SA 4.0" --max-car 5000000 --sortie data
Les sources sont des fichiers texte LOCAUX : ce script ne télécharge rien.
"""
import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter

INCONNU = "¤"


def normaliser(t):
    t = unicodedata.normalize("NFC", t).replace("\r\n", "\n").replace("\r", "\n")
    t = t.translate({0x2019: "'", 0x2018: "'", 0x00A0: " ", 0x202F: " ", 0x2009: " ", 0x00AD: None, 0xFEFF: None})
    t = "".join(c for c in t if c == "\n" or unicodedata.category(c)[0] != "C")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t)


def documents(textes, taille):
    """Regroupe les paragraphes consécutifs en documents d'environ `taille` caractères,
    en retirant les paragraphes déjà vus (doublons exacts, ≥ 60 caractères)."""
    vus, docs, retires = set(), [], 0
    for t in textes:
        courant, n = [], 0
        for p in t.split("\n\n"):
            p = p.strip()
            if not p:
                continue
            if len(p) >= 60:
                h = hashlib.sha256(p.lower().encode("utf-8")).digest()
                if h in vus:
                    retires += 1
                    continue
                vus.add(h)
            courant.append(p)
            n += len(p) + 2
            if n >= taille:
                docs.append("\n\n".join(courant) + "\n\n")
                courant, n = [], 0
        if courant:
            docs.append("\n\n".join(courant) + "\n\n")
    return docs, retires


def sha(chemin):
    return hashlib.sha256(open(chemin, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--licence", required=True)
    ap.add_argument("--max-car", type=int, default=5_000_000)
    ap.add_argument("--taille-doc", type=int, default=2000)
    ap.add_argument("--min-compte", type=int, default=20, help="caractères plus rares dans le train → ¤")
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sortie", default="data")
    a = ap.parse_args()
    d = os.path.join(a.sortie, a.nom)
    if os.path.exists(os.path.join(d, "manifest.json")):
        raise SystemExit(f"{d} existe déjà : un corpus publié ne se réécrit pas, changer --nom")

    docs, retires = documents([normaliser(open(s, encoding="utf-8").read()) for s in a.sources], a.taille_doc)
    random.Random(a.seed).shuffle(docs)
    garde, n = [], 0
    for doc in docs:
        if n >= a.max_car:
            break
        garde.append(doc)
        n += len(doc)
    if len(garde) < 20:
        raise SystemExit(f"Seulement {len(garde)} documents : corpus trop petit pour un découpage train/val/test fiable")

    test, val, train, nt, nv = [], [], [], 0, 0
    for doc in garde:                                  # déjà mélangés : remplissage test, puis val, puis train
        if nt < a.test * n:
            test.append(doc); nt += len(doc)
        elif nv < 0.10 * (1 - a.test) * n:
            val.append(doc); nv += len(doc)
        else:
            train.append(doc)
    tr, va, te = "".join(train), "".join(val), "".join(test)
    # ajustement exact à 90/10 pour int(0.9*len(corpus)) : on rogne la fin du côté en excès
    cible = len(va) * 9
    if len(tr) >= cible:
        tr = tr[len(tr) - cible:]
    else:
        va = va[:len(tr) // 9]
        tr = tr[:len(va) * 9]

    gardes = {c for c, k in Counter(tr).items() if k >= a.min_compte}
    rempl = {}
    for nom, t in (("train", tr), ("val", va), ("test", te)):
        rempl[nom] = sum(c not in gardes for c in t)
    f = lambda t: "".join(c if c in gardes else INCONNU for c in t)
    tr, va, te = f(tr), f(va), f(te)
    assert int(0.9 * len(tr + va)) == len(tr), "la coupe 90/10 de hybrid_numpy ne tombe pas sur la frontière train/val"

    os.makedirs(d, exist_ok=True)
    for nom, t in (("corpus.txt", tr + va), ("test.txt", te)):
        with open(os.path.join(d, nom), "w", encoding="utf-8", newline="\n") as g:
            g.write(t)
    man = {"nom": a.nom, "licence": a.licence, "seed": a.seed,
           "sources": [{"fichier": os.path.basename(s), "sha256": sha(s)} for s in a.sources],
           "caracteres": {"train": len(tr), "val": len(va), "test": len(te)},
           "documents": {"train": len(train), "val": len(val), "test": len(test)},
           "paragraphes_dupliques_retires": retires,
           "vocabulaire": len(set(tr + va + te)), "remplaces_par_inconnu": rempl,
           "sha256": {n_: sha(os.path.join(d, n_)) for n_ in ("corpus.txt", "test.txt")}}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
