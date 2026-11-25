#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C8 — budget d'entraînement sur data/fr1m (pensé pour GitHub Actions, marche aussi en local).

Même boucle que scripts/c3b_controles.py (entrainer) : même init, mêmes batchs, même optimiseur, cosinus réglé sur
--max-steps. Ajouts : intervalle d'évaluation, pas des fenêtres de val, meilleur checkpoint (val entière minimale),
temps par point, et évaluation du TEST une seule fois, à la fin (poids finaux ET meilleurs poids).
Rien n'est choisi avec le test. Les scripts existants ne sont pas modifiés.

  python3 scripts/c8_github.py --controle-c7 --variante smax          (rejoue 300 pas de C7 et compare)
  python3 scripts/c8_github.py --variante dlind-const --seed 611 --max-steps 5400 --out-dir results/c8g_dlind-const_b5400_s611
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time

RACINE = os.environ.get("IMANE_RACINE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "experiments"))
sys.path.insert(0, os.path.join(RACINE, "scripts"))
import numpy as np  # noqa: E402
import hybrid_numpy as hn  # noqa: E402
import c3b_controles as cc  # noqa: E402

VARIANTES = ("smax", "dlind-elu1", "dlind-const")
BUDGETS = (18000, 5400, 1800)
SEEDS = (611, 622, 633, 644, 655)
T, LOT, D, TETES, INCONNU, MIN_COMPTE = 96, 16, 96, 4, "¤", 20


def fenetres(val, pas):
    """Comme c3b_controles.fenetres (pas 16 = identique), avec un pas réglable."""
    d = list(range(0, len(val) - T - 1, pas)) + [len(val) - T - 1]
    return np.stack([val[i:i + T] for i in d]), np.stack([val[i + 1:i + T + 1] for i in d])


def intervalle_eval(budget):
    """Environ 30 points par run : 100 pas à 1 800, 200 à 5 400, 600 à 18 000."""
    return max(100, int(round(budget / 30 / 100.0)) * 100)


def ids_texte(texte, stoi, gardes):
    """Caractères inconnus ou rares (compte < 20 dans le train) remplacés par ¤ et comptés (règle de evaluer_test.py)."""
    if INCONNU not in stoi:
        raise SystemExit("le caractère ¤ manque dans le vocabulaire du corpus")
    n, ids = 0, []
    for c in texte:
        if c not in gardes:
            n, c = n + 1, INCONNU
        ids.append(stoi[c])
    return np.array(ids, dtype=np.int32), n


def infos_env():
    def git(*a):
        r = subprocess.run(["git", *a], cwd=RACINE, capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip()
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for ligne in f:
                if ligne.startswith("model name"):
                    cpu = ligne.split(":", 1)[1].strip()
                    break
    return {"commit": git("rev-parse", "HEAD"),
            "code_modifie": bool(git("status", "--porcelain", "--", "experiments", "scripts", "data")),
            "python": sys.version.split()[0], "numpy": np.__version__, "machine": platform.platform(), "cpu": cpu,
            "coeurs": os.cpu_count(), "github_run": os.environ.get("GITHUB_RUN_ID", ""),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def preparer(variante, chemin):
    types, phi, nk, delta = cc.VARIANTES[variante]
    cc.installer(phi, nk, delta)
    _, ids, stoi, itos = hn.charger_donnees(chemin)
    n = int(0.9 * len(ids))
    cfg = {"vocab_size": len(itos), "d_model": D, "n_head": TETES, "types": types.split(","), "block_size": T}
    return (types, phi, nk, delta), ids, n, stoi, itos, cfg


def entrainer(a, arret_apres=None, suivi=None):
    """arret_apres / suivi : réservés au contrôle C7 (arrêt avant la fin, aucune sortie de résultat)."""
    (types, phi, nk, delta), ids, n, stoi, itos, cfg = preparer(a.variante, a.data)
    tr, va = ids[:n], ids[n:]
    Xv, Yv = fenetres(va, a.pas_eval)
    rng = np.random.default_rng(a.seed)
    P = hn.init_params(len(itos), cfg, rng)
    opt = hn.AdamW(P, wd=0.05)
    os.makedirs(a.out_dir, exist_ok=True)
    h, t0, meilleur, meilleurs, pas_best = [], time.time(), float("inf"), None, 0
    for pas in range(a.max_steps + 1):
        lr = hn.taux_apprentissage(pas, 1e-3, 100, a.max_steps)
        x, y = hn.prendre_batch(tr, LOT, T, rng)
        _, perte, cache = hn.forward(P, cfg, x, y)
        g = hn.backward(P, cfg, cache, y)
        hn.clip_gradient(g, 1.0)
        opt.step(g, lr)
        if pas % a.eval_every == 0 or pas == a.max_steps:
            pv = float(np.mean([hn.forward(P, cfg, *hn.prendre_batch(va, LOT, T, rng))[1] for _ in range(10)]))
            pe = cc.eval_entiere(P, cfg, Xv, Yv)
            h.append((pas, float(perte), pv, pe, time.time() - t0))
            print(f"{pas:>6} | train {float(perte):.4f} | val {pv:.4f} | val entière {pe:.4f} | {time.time() - t0:.0f}s", flush=True)
            with open(os.path.join(a.out_dir, "historique.csv"), "w", encoding="utf-8", newline="\n") as f:
                f.write("step,perte_train,perte_val,perte_val_entiere,temps_s\n")
                f.writelines(",".join(map(str, r)) + "\n" for r in h)
            if pe < meilleur:
                meilleur, pas_best = pe, pas
                meilleurs = {k: v.copy() for k, v in P.items()}
        if arret_apres is not None and pas >= arret_apres:
            break
    if suivi is not None:
        suivi.extend(h)
        return None
    np.savez_compressed(os.path.join(a.out_dir, "checkpoint.npz"), **P)
    np.savez_compressed(os.path.join(a.out_dir, "checkpoint_best.npz"), **meilleurs)

    # ---- TEST : une seule évaluation par jeu de poids, à la fin ----
    compte = np.bincount(tr, minlength=len(itos))
    gardes = {itos[i] for i in range(len(itos)) if compte[i] >= MIN_COMPTE}
    res_test = {"variante": a.variante, "seed": a.seed, "budget": a.max_steps, "pas_best": pas_best}
    for nom, chemin in (("test", a.test), ("oeuvres", a.oeuvres)):
        if not chemin:
            continue
        with open(chemin, encoding="utf-8") as f:
            tids, inc = ids_texte(f.read(), stoi, gardes)
        for lib, poids in (("final", P), ("best", meilleurs)):
            res_test[f"{nom}_{lib}"] = float(hn.evaluer_fixe(poids, cfg, tids, T))
        res_test[f"{nom}_caracteres"] = int(len(tids))
        res_test[f"{nom}_inconnus"] = int(inc)
    with open(os.path.join(a.out_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(res_test, f, indent=1, ensure_ascii=False)

    ve = [r[3] for r in h]
    res = {"variante": a.variante, "types": types.split(","), "phi": phi, "cles_normalisees": nk, "delta": delta, "seed": a.seed,
           "budget": a.max_steps, "eval_every": a.eval_every, "pas_eval": a.pas_eval,
           "corpus": os.path.basename(os.path.dirname(os.path.abspath(a.data))),
           "params": sum(v.size for v in P.values()), "val_entiere_best": meilleur, "pas_best": pas_best,
           "val_entiere_min": min(ve), "val_entiere_final": ve[-1], "train_final": h[-1][1], "temps_total_s": h[-1][4],
           "secondes_par_pas": h[-1][4] / max(1, a.max_steps)}
    with open(os.path.join(a.out_dir, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("RÉSUMÉ :", json.dumps({**res, **res_test}, ensure_ascii=False))
    return res


def controle_c7(variante, racine_c7, tolerance, pas_max):
    """Rejoue les premiers pas du run C7 seed 101 (mêmes réglages) et compare à historique.csv."""
    with open(os.path.join(racine_c7, f"c7a_{variante}_s101", "historique.csv"), encoding="utf-8") as f:
        ref = list(csv.DictReader(f))
    sortie = os.path.join(RACINE, "results", "_controle_c8")
    a = argparse.Namespace(variante=variante, seed=101, max_steps=1800, eval_every=100, pas_eval=16,
                           data=os.path.join(RACINE, "data", "fr1m", "corpus.txt"), out_dir=sortie)
    suivi = []
    entrainer(a, arret_apres=pas_max, suivi=suivi)
    shutil.rmtree(sortie, ignore_errors=True)
    pire = 0.0
    for (pas, ptr, pv, pe, _), r in zip(suivi, ref):
        if int(r["step"]) != pas:
            raise SystemExit("les pas ne correspondent pas")
        pire = max(pire, abs(ptr - float(r["perte_train"])), abs(pv - float(r["perte_val"])), abs(pe - float(r["perte_val_entiere"])))
    ok = len(suivi) >= pas_max // 100 + 1 and pire < tolerance
    print(f"{'OK' if ok else 'ÉCHEC'} : contrôle C7 {variante}, {len(suivi)} points, pire écart {pire:.2e} (tolérance {tolerance:g})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=VARIANTES)
    ap.add_argument("--seed", type=int, default=611)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = automatique (environ 30 points)")
    ap.add_argument("--pas-eval", type=int, default=16, help="pas entre fenêtres de val (16 = C7)")
    ap.add_argument("--data", default=os.path.join(RACINE, "data", "fr1m", "corpus.txt"))
    ap.add_argument("--test", default=os.path.join(RACINE, "data", "fr1m", "test.txt"))
    ap.add_argument("--oeuvres", default=os.path.join(RACINE, "data", "test_oeuvres_v2.txt"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--controle-c7", action="store_true", help="vérifie que ce script refait C7 (300 pas)")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--pas-controle", type=int, default=300)
    a = ap.parse_args()
    if a.controle_c7:
        sys.exit(0 if controle_c7(a.variante, os.path.join(RACINE, "results"), a.tolerance, a.pas_controle) else 1)
    if not a.variante or not a.out_dir:
        raise SystemExit("--variante et --out-dir sont obligatoires (ou --controle-c7)")
    if os.path.exists(os.path.join(a.out_dir, "resume.json")):
        raise SystemExit(f"{a.out_dir} contient déjà un run terminé (règle 4 : on n'écrase pas)")
    if a.eval_every <= 0:
        a.eval_every = intervalle_eval(a.max_steps)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "ENV.json"), "w", encoding="utf-8") as f:
        json.dump(infos_env(), f, indent=1, ensure_ascii=False)
    entrainer(a)


if __name__ == "__main__":
    main()
