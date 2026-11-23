#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur parallèle local : remplace les boucles séquentielles run_c2.sh / run_c3.sh.

NumPy n'utilise plusieurs cœurs que dans les gros produits matriciels : sur 2 cœurs,
deux processus à 1 thread vont nettement plus vite qu'un processus à 2 threads.

  python3 scripts/lancer_parallele.py --cycle c5 --configs "lin,lin" "dlind,dlind" --seeds 1337 2024 777
  python3 scripts/lancer_parallele.py --cycle c3b --variantes --configs smax lin-elu1 dlind-const

Garanties :
 - n'écrase jamais un run terminé (règle 4 du protocole : baselines figées) ; un run
   interrompu bloque le lancement, sauf avec --reprendre (il repart alors de zéro et
   remplace son propre contenu incomplet) ;
 - priorité processus basse (la machine reste utilisable) ;
 - trace le commit, l'état « dirty » et l'environnement de chaque run.
"""
import argparse
import json
import os
import platform
import subprocess
import shlex
import sys
import time

def racine_depot():
    """Remonte jusqu'au dossier qui contient experiments/hybrid_numpy.py (marche depuis scripts/ ou propositions/scripts/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "experiments", "hybrid_numpy.py")):
        p = os.path.dirname(d)
        if p == d:
            return os.getcwd()
        d = p
    return d


RACINE = racine_depot()


def git(*a, cwd):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="préfixe des dossiers (ex. c3b)")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2024, 777])
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--jobs", type=int, default=2, help="processus simultanés (≈ nombre de cœurs physiques)")
    ap.add_argument("--racine", default=RACINE, help="racine du dépôt")
    ap.add_argument("--sortie", default="", help="dossier des résultats (défaut : <racine>/results)")
    ap.add_argument("--rapide", default="", help="chemin de moteur_rapide.py (optionnel)")
    ap.add_argument("--extra", default="", help='options supplémentaires transmises au script, ex. --extra "--phi elu1 --ternary"')
    ap.add_argument("--variantes", action="store_true",
                    help="lance c3b_controles.py --variante (même dossier) au lieu de hybrid_numpy.py --types")
    ap.add_argument("--data", default="", help="corpus (défaut : data/corpus_fr.txt)")
    ap.add_argument("--reprendre", action="store_true")
    a = ap.parse_args()
    if not a.cycle.replace("-", "").isalnum():
        raise SystemExit("--cycle : lettres, chiffres et tirets uniquement")
    sortie = os.path.abspath(a.sortie or os.path.join(a.racine, "results"))
    script = os.path.join(a.racine, "experiments", "hybrid_numpy.py")

    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    prep, drapeau = ["-u", script, "--train"], "--types"
    if a.variantes:
        prep = ["-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "c3b_controles.py")]
        drapeau = "--variante"
        env["IMANE_RACINE"] = a.racine
    elif a.rapide:
        code = ("import sys,os; sys.path[:0]=[%r,%r]; import hybrid_numpy as hn, moteur_rapide as m; "
                "m.installer(hn); hn.main()") % (os.path.dirname(script), os.path.dirname(os.path.abspath(a.rapide)))
        prep = ["-u", "-c", code, "--train"]
    extra = shlex.split(a.extra)
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        options["preexec_fn"] = lambda: os.nice(10)

    file, sautes = [], []
    for seed in a.seeds:                      # seed par seed : un cycle interrompu reste apparié
        for cfg in a.configs:
            tag = f"{a.cycle}_{cfg.replace(',', '-')}_s{seed}"
            d = os.path.join(sortie, tag)
            if os.path.exists(os.path.join(d, "resume.json")):
                sautes.append(tag + " (déjà terminé)")
                continue
            if os.path.isdir(d) and os.listdir(d) and not a.reprendre:
                raise SystemExit(f"{d} existe et n'est pas terminé : relancer avec --reprendre, ou changer --cycle")
            file.append((tag, d, cfg, seed))
    for s in sautes:
        print("sauté :", s)

    info = {"commit": git("rev-parse", "HEAD", cwd=a.racine),
            "code_modifie": bool(git("status", "--porcelain", "--", "experiments", cwd=a.racine)),
            "python": sys.version.split()[0], "machine": platform.platform(), "moteur": a.rapide or "reference"}
    if info["code_modifie"]:
        print("ATTENTION : experiments/ contient des modifications non commitées (protocole étape 3).")
    actifs, debut, rates = [], time.time(), 0
    while file or actifs:
        while file and len(actifs) < a.jobs:
            tag, d, cfg, seed = file.pop(0)
            os.makedirs(d, exist_ok=True)
            json.dump(info, open(os.path.join(d, "ENV.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            log = open(os.path.join(d, "train.log"), "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, *prep, drapeau, cfg, "--seed", str(seed),
                                  "--max-steps", str(a.max_steps), "--out-dir", d, *extra,
                                  "--data", a.data or os.path.join(a.racine, "data", "corpus_fr.txt")],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=a.racine, **options)
            actifs.append((p, tag, log, time.time()))
            print(f"[{time.time() - debut:6.0f}s] lancé   {tag}", flush=True)
        time.sleep(0.5)
        for x in list(actifs):
            p, tag, log, t0 = x
            if p.poll() is not None:
                log.close()
                actifs.remove(x)
                rates += p.returncode != 0
                print(f"[{time.time() - debut:6.0f}s] {'terminé' if p.returncode == 0 else 'ÉCHEC  '} {tag} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"Fin : {time.time() - debut:.0f}s, {rates} échec(s). Analyse : "
          f"python scripts/agreger.py --prefixes {a.cycle}")
    sys.exit(1 if rates else 0)


if __name__ == "__main__":
    main()
