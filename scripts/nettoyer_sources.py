#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nettoyage déterministe des sources brutes (wikisource), sans accès réseau.

Entrée : un fichier .txt par œuvre (sources brutes hors dépôt), paragraphes séparés
par une ligne vide.
Sortie : un fichier .txt nettoyé par œuvre, mêmes séparations, UTF-8 NFC.

Règles de nettoyage (appliquées dans cet ordre, paragraphe par paragraphe) :
  1. Unicode NFC ;
  2. re-collage des lignes coupées à l'intérieur d'un paragraphe (une ligne qui se
     termine par « - » fusionne sans espace) ;
  3. suppression des paragraphes non-prose : titres de chapitre en capitales,
     numéros de page/romains, ligne d'ornement (* * *, ⁂, ---), marqueurs
     d'illustration/notes [ … ], en-têtes de transcription ;
  4. suppression des paragraphes contenant http ou @ (contrôle du protocole) ;
  5. plafond par auteur (--max-par-auteur) pour la contrainte « aucun auteur
     au-dessus de 20 % du corpus » : les œuvres sont ajoutées dans l'ordre donné,
     la dernière est tronquée à la frontière de paragraphe.

Les textes bruts et nettoyés restent hors du dépôt ; les empreintes SHA-256 des
fichiers nettoyés sont consignées dans data/SOURCES.md (généré à part).
"""
import argparse
import os
import re
import unicodedata

ROMAIN = re.compile(r"^[MDCLXVIVX]+\.?$")
ORNEMENT = re.compile(r"^[\s\*⁂·—–\-_=~]+$")
CAPITALES = re.compile(r"^[^a-zà-öø-ÿ]*$")
NUMERO = re.compile(r"^\d{1,4}$")


def est_artefact(p):
    """Paragraphe qui n'est pas de la prose : titre en capitales, numéro,
    ornement, illustration, mention de transcription."""
    s = p.strip()
    if not s:
        return True
    lettres = [c for c in s if c.isalpha()]
    if lettres and sum(c.isupper() for c in lettres) / len(lettres) > 0.6 and len(lettres) <= 80:
        return True                                   # titre de chapitre en capitales
    if NUMERO.match(s) or ROMAIN.match(s):
        return True                                   # numéro de page / partie
    if ORNEMENT.match(s):
        return True                                   # ligne d'ornement
    if s.startswith("[") or s.startswith("⁂") or s.startswith("(illustration"):
        return True
    if re.search(r"https?://|www\.|@", s):
        return True                                   # trace de site / courriel
    if re.search(r"[Tt]ranscription|[Ww]ikisource|[Pp]rojet [Gg]utenberg|[Éé]dition [Éée]lectronique|source numérique", s[:120]):
        return True                                   # mention de transcription
    return False


def nettoyer_texte(t):
    """Retourne la liste des paragraphes propres d'une œuvre."""
    t = unicodedata.normalize("NFC", t)
    paras = [b.strip() for b in t.split("\n\n") if b.strip()]
    propres = []
    for bloc in paras:
        lignes = [l.strip() for l in bloc.split("\n") if l.strip()]
        # re-collage : une ligne finissant par « - » fusionne sans espace
        rassemble = ""
        for l in lignes:
            if rassemble.endswith("-"):
                rassemble = rassemble[:-1] + l
            else:
                rassemble = (rassemble + " " + l) if rassemble else l
        if not est_artefact(rassemble):
            propres.append(rassemble)
    return propres


def main():
    ap = argparse.ArgumentParser(description="Nettoyage déterministe des sources brutes")
    ap.add_argument("--brut", required=True, help="dossier des textes bruts (hors dépôt)")
    ap.add_argument("--sortie", required=True, help="dossier des textes nettoyés (hors dépôt)")
    ap.add_argument("--max-par-auteur", type=int, default=1_000_000,
                    help="plafond de caractères nettoyés par auteur (contrôle 20 %% du corpus)")
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)

    # prélevés : fichiers triés, préfixe du nom = auteur
    fichiers = sorted(f for f in os.listdir(a.brut) if f.endswith(".txt"))
    par_auteur = {}
    for f in fichiers:
        auteur = f.split("_")[0]
        par_auteur.setdefault(auteur, []).append(f)

    # nettoyage une fois pour toutes, puis répartition PROPORTIONNELLE du plafond
    # par auteur entre ses œuvres (aucune œuvre réduite à zéro)
    nettoyes = {f: nettoyer_texte(open(os.path.join(a.brut, f), encoding="utf-8").read())
                for f in fichiers}
    total = 0
    rapport = []
    for auteur, fs in sorted(par_auteur.items()):
        longueurs = {f: sum(len(p) + 2 for p in nettoyes[f]) for f in fs}
        somme = sum(longueurs.values())
        echelle = min(1.0, a.max_par_auteur / max(somme, 1))
        for f in fs:
            budget_f = int(longueurs[f] * echelle)
            garde = []
            n = 0
            for p in nettoyes[f]:
                if n >= budget_f:
                    break
                garde.append(p)
                n += len(p) + 2
            chemin = os.path.join(a.sortie, f)
            with open(chemin, "w", encoding="utf-8", newline="\n") as g:
                g.write("\n\n".join(garde) + "\n")
            total += n
            rapport.append((f, n, len(nettoyes[f]), len(garde)))
            print(f"{f:55s} {n:>10,} car. ({len(garde)}/{len(nettoyes[f])} § gardés)")
    print(f"TOTAL nettoyé : {total:,} car. — {len(fichiers)} œuvres, {len(par_auteur)} auteurs")


if __name__ == "__main__":
    main()
