#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrôle automatique d'un corpus préparé (data/<nom>/).

Vérifie, pour data/<nom>/ (corpus.txt, test.txt, manifest.json) :
  1. manifest.json lisible ; SHA-256 de corpus.txt et test.txt correspondent
     (fichiers en fins de ligne LF — voir .gitattributes data/** -text) ;
  2. vocabulaire dans [60, 160] ; part de ¤ < 0.1 % ; tout caractère du test est
     dans l'alphabet du corpus (ou remplacé par ¤) ;
  3. copies : 0 fenêtre de 60 caractères (pas de 5) de la val ou du test présente
     mot pour mot dans le train (critère C8/étape 1) ;
  4. test nettoyé comme le corpus : pas d'apostrophe courbe ’, pas de trace
     (http, @, Project Gutenberg, ***), longueur moyenne des paragraphes en
     [150, 1500].
Échec = code retour 1 avec la liste des échecs.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter

INCONNU = "¤"


def sha_lf(chemin):
    b = open(chemin, "rb").read()
    if b"\r" in b:
        return None, "fins de ligne non-LF (CRLF)"
    return hashlib.sha256(b).hexdigest(), None


def fenetres_dupliques(train, t, L=60, pas=5):
    idx = {train[i:i + L] for i in range(0, len(train) - L + 1)}
    vus, n = set(), 0
    for i in range(0, len(t) - L + 1, pas):
        w = t[i:i + L]
        if w in idx and w not in vus:
            vus.add(w)
            n += len(vus)
    return len(vus)


def main():
    ap = argparse.ArgumentParser(description="Vérifie un corpus préparé")
    ap.add_argument("--dossier", required=True, help="data/<nom>")
    a = ap.parse_args()
    d = a.dossier
    echecs = []

    corpus = open(f"{d}/corpus.txt", encoding="utf-8").read()
    test = open(f"{d}/test.txt", encoding="utf-8").read()
    man = json.load(open(f"{d}/manifest.json"))
    n_train = int(0.9 * len(corpus))
    train, val = corpus[:n_train], corpus[n_train:]

    # 1. manifest + SHA-256 (LF)
    for nom in ("corpus.txt", "test.txt"):
        s, err = sha_lf(f"{d}/{nom}")
        if err:
            echecs.append(f"SHA {nom} : {err}")
        elif s != man["sha256"][nom]:
            echecs.append(f"SHA {nom} ≠ manifest")

    # 2. vocabulaire + inconnus
    alpha_train = set(train)
    alpha_test = set(test)
    vocab = len(set(corpus) | set(test))
    part_inconnu = (corpus + test).count(INCONNU) / (len(corpus) + len(test))
    hors_train = sorted(c for c in alpha_test if c not in alpha_train and c != INCONNU)
    if not (60 <= vocab <= 160):
        echecs.append(f"vocabulaire {vocab} hors [60, 160]")
    if part_inconnu >= 0.001:
        echecs.append(f"part de ¤ = {part_inconnu * 100:.3f} % ≥ 0.1 %")
    if hors_train:
        echecs.append(f"caractères du test absents du corpus : {hors_train}")

    # 3. copies train ↔ val/test
    nv = fenetres_dupliques(train, val)
    nt = fenetres_dupliques(train, test)
    if nv or nt:
        echecs.append(f"fenêtres 60 car. dupliquées : val {nv}, test {nt} (critère : 0)")
    print(f"[copies] val {nv} | test {nt} (fenêtres uniques de 60 car., pas 5, présentes dans le train)")

    # 4. test nettoyé comme le corpus
    traces = {s: test.count(s) for s in ("http", "@", "Project Gutenberg", "***")}
    if traces["http"] or traces["@"] or traces["Project Gutenberg"] or traces["***"]:
        echecs.append(f"traces dans le test : {traces}")
    if "’" in test:
        echecs.append(f"apostrophes courbes dans le test : {test.count(chr(0x2019))} (normaliser attend ’ → ')")
    paras = [p for p in test.split("\n\n") if p.strip()]
    moy = sum(len(p) for p in paras) / max(len(paras), 1)
    if not (150 <= moy <= 1500):
        echecs.append(f"longueur moyenne des paragraphes du test : {moy:.0f} hors [150, 1500]")

    print(f"[manifest] SHA-256 ✔ | vocab {vocab} | ¤ {part_inconnu * 100:.4f} % | "
          f"§ test {len(paras)} (moy. {moy:.0f} car.) | ’ : {test.count(chr(0x2019))}")
    if echecs:
        print("ÉCHECS :")
        for e in echecs:
            print("  -", e)
        sys.exit(1)
    print("CORPUS CONFORME ✔")


if __name__ == "__main__":
    main()
