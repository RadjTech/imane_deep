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
  5. plafond par auteur (--max-par-auteur), réparti proportionnellement entre les
     œuvres d'un même auteur, pour la contrainte « aucun auteur au-dessus de 20 % ».

--anti-repets (C8, correctif du défaut 2) : dédoublonnage par fenêtre glissante.
Tout paragraphe déjà vu textuellement (exact, ≥ 15 car.) ou contenant une fenêtre
de 60 caractères déjà présente dans le texte conservé est retiré. Les fenêtres ne
couvrant QUE la queue (aucun caractère du paragraphe candidat) ne sont ni
vérifiées ni indexées — les indexer provoquait un verrouillage (rejet systématique
de tous les paragraphes suivants dès que la queue réapparaissait).
Les textes bruts et nettoyés restent hors du dépôt.
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
    """Retourne la liste des paragraphes propres d'une œuvre (mode simple)."""
    t = unicodedata.normalize("NFC", t)
    paras = [b.strip() for b in t.split("\n\n") if b.strip()]
    propres = []
    for bloc in paras:
        lignes = [l.strip() for l in bloc.split("\n") if l.strip()]
        rassemble = ""
        for l in lignes:
            if rassemble.endswith("-"):
                rassemble = rassemble[:-1] + l
            else:
                rassemble = (rassemble + " " + l) if rassemble else l
        if not est_artefact(rassemble):
            propres.append(rassemble)
    return propres


def nettoyer_anti_repets(brut, fichiers, rapport):
    """Mode --anti-repets : retourne {fichier: [paragraphes gardés]}.
    Index global des fenêtres de 60 caractères (tous offsets) du texte conservé,
    en ne considérant que les fenêtres contenant du contenu nouveau."""
    import hashlib
    from collections import Counter

    import numpy as np

    def dig(ch):
        return int.from_bytes(hashlib.blake2b(ch.encode("utf-8"),
                                              digest_size=8).digest(), "big") & ((1 << 63) - 1)

    def digests_region(chaine, i, j):
        if j - i < 60:
            return []
        return [dig(chaine[k:k + 60]) for k in range(i, j - 59)]

    maitre = np.empty(0, dtype=np.int64)
    tampon, tampon_set = [], set()
    vus_paras = set()
    queue = ""
    nettoyes = {}
    for f in fichiers:
        t = unicodedata.normalize("NFC", open(os.path.join(brut, f), encoding="utf-8").read())
        paras = [b.strip() for b in t.split("\n\n") if b.strip()]
        garde = []
        rejets = Counter()
        for bloc in paras:
            lignes = [l.strip() for l in bloc.split("\n") if l.strip()]
            rassemble = ""
            for l in lignes:
                if rassemble.endswith("-"):
                    rassemble = rassemble[:-1] + l
                else:
                    rassemble = (rassemble + " " + l) if rassemble else l
            p = rassemble
            if est_artefact(p):
                rejets["artefact"] += 1
                continue
            cle = re.sub(r"\s+", " ", p).lower()
            if len(cle) >= 15 and cle in vus_paras:
                rejets["paragraphe répété"] += 1
                continue
            region = (queue + "\n\n" + p) if queue else p
            k_depart = max(0, len(queue) - 57)   # fenêtres contenant du contenu nouveau
            nouveaux = digests_region(region, k_depart, len(region))
            en_double = False
            for d in nouveaux:
                k = np.searchsorted(maitre, d)
                if (k < len(maitre) and maitre[k] == d) or d in tampon_set:
                    en_double = True
                    break
            if en_double:
                rejets["fenêtre répétée"] += 1
                continue
            if len(cle) >= 15:
                vus_paras.add(cle)
            if len(tampon) > 400_000:
                maitre = np.unique(np.concatenate([maitre, np.array(tampon, dtype=np.int64)]))
                tampon, tampon_set = [], set()
            tampon.extend(nouveaux)
            tampon_set.update(nouveaux)
            queue = region[-59:]
            garde.append(p)
        nettoyes[f] = garde
        rapport.append((f, len(garde), len(paras), dict(rejets)))
        print(f"  {f}: {len(garde)}/{len(paras)} § gardés ({dict(rejets)})", flush=True)
    if tampon:
        maitre = np.unique(np.concatenate([maitre, np.array(tampon, dtype=np.int64)]))
    return nettoyes, len(maitre)


def main():
    ap = argparse.ArgumentParser(description="Nettoyage déterministe des sources brutes")
    ap.add_argument("--brut", required=True, help="dossier des textes bruts (hors dépôt)")
    ap.add_argument("--sortie", required=True, help="dossier des textes nettoyés (hors dépôt)")
    ap.add_argument("--max-par-auteur", type=int, default=1_000_000,
                    help="plafond de caractères nettoyés par auteur (contrôle 20 %% du corpus)")
    ap.add_argument("--anti-repets", action="store_true",
                    help="dédoublonnage par fenêtre glissante (correctif défaut 2)")
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)

    fichiers = sorted(f for f in os.listdir(a.brut) if f.endswith(".txt"))
    par_auteur = {}
    for f in fichiers:
        par_auteur.setdefault(f.split("_")[0], []).append(f)

    if a.anti_repets:
        nettoyes, n_fenetres = nettoyer_anti_repets(a.brut, fichiers, [])
    else:
        nettoyes = {f: nettoyer_texte(open(os.path.join(a.brut, f), encoding="utf-8").read())
                    for f in fichiers}

    # répartition proportionnelle du plafond par auteur entre ses œuvres
    total = 0
    for auteur, fs in sorted(par_auteur.items()):
        longueurs = {f: sum(len(p) + 2 for p in nettoyes[f]) for f in fs}
        somme = sum(longueurs.values())
        echelle = min(1.0, a.max_par_auteur / max(somme, 1))
        for f in fs:
            budget_f = int(longueurs[f] * echelle)
            garde, n = [], 0
            for p in nettoyes[f]:
                if n >= budget_f:
                    break
                garde.append(p)
                n += len(p) + 2
            with open(os.path.join(a.sortie, f), "w", encoding="utf-8", newline="\n") as g:
                g.write("\n\n".join(garde) + "\n")
            total += n
            print(f"{f:55s} {n:>10,} car. ({len(garde)}/{len(nettoyes[f])} § gardés)")
    print(f"TOTAL nettoyé : {total:,} car. — {len(fichiers)} œuvres, {len(par_auteur)} auteurs"
          + (f" | fenêtres indexées : {n_fenetres:,}" if a.anti_repets else ""))


if __name__ == "__main__":
    main()
