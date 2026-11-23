#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 (partie 2) — Profillage d'inférence batch-1 des trois mécanismes.

Mesures :
  1. GARDE-FOU : équivalence incrémental (état lin / KV-cache smax) ↔ forme
     parallèle (hybrid_numpy.forward) sur un contexte de 96 tokens.
     Seuil : écart max sur les logits < 2e-3 (float32).
  2. V_gen (tok/s, batch 1) mesuré après absorption d'un prompt de T0 tokens,
     T0 ∈ {16, 48, 96} — génération gloutonne de 150 tokens.
  3. Contraste naïf (re-calcul complet de la fenêtre à chaque token) pour
     smax — c'est la différence entre « sans optimisation » et KV-cache.
  4. Scaling mécanique : pos_emb neutralisée, absorption de T ∈ {256, 1024,
     4096} tokens puis génération de 100 tokens → V_gen en fonction de la
     longueur du contexte (hors distribution QUALITÉ, mesure du mécanisme).
  5. Mémoire : poids, état linéaire (constant), KV-cache (croît avec T).

Usage :
  python3 experiments/profiler_inf.py --ckpt results/c1_hybride_lin_smax/checkpoint.npz --types lin,smax
  python3 experiments/profiler_inf.py --ckpt results/c2_smax-smax_s1337/checkpoint.npz --types smax,smax
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_numpy as hn  # noqa: E402


def charger_corpus(chemin):
    texte = open(chemin, encoding="utf-8").read()
    chars = sorted(set(texte))
    stoi = {c: i for i, c in enumerate(chars)}
    return np.array([stoi[c] for c in texte], dtype=np.int32), chars


class GenerationIncrementale:
    """Un token à la fois : lin → état constant ; smax → KV-cache incrémental."""

    def __init__(self, P, types, n_head, Tmax=4096):
        self.P, self.types, self.n_head = P, types, n_head
        self.D = P["tok_emb"].shape[1]
        self.dh = self.D // n_head
        self.Tmax = Tmax
        self.scale = 1.0 / math.sqrt(self.dh)
        self.reset()

    def reset(self):
        P, D, H, dh = self.P, self.D, self.n_head, self.dh
        self.S, self.Z, self.kv = {}, {}, {}
        for l, t in enumerate(self.types):
            if t == "lin":
                self.S[l] = np.zeros((H, dh, dh), np.float32)
                self.Z[l] = np.zeros((H, dh), np.float32)
            else:
                self.kv[l] = [np.zeros((self.Tmax, H, dh), np.float32),
                              np.zeros((self.Tmax, H, dh), np.float32), 0]
        self.t = 0

    def pas(self, tok):
        P, D, H, dh = self.P, self.D, self.n_head, self.dh
        pos = min(self.t, P["pos_emb"].shape[0] - 1)
        x = P["tok_emb"][int(tok)][None, :] + P["pos_emb"][pos][None, :]
        for l, typ in enumerate(self.types):
            h1, _, _ = hn.layer_norm_fw(x, P[f"b{l}.ln1.g"], P[f"b{l}.ln1.b"])
            qkv = h1 @ P[f"b{l}.qkv.W"] + P[f"b{l}.qkv.b"]
            q = qkv[:, :D].reshape(H, dh)
            k = qkv[:, D:2 * D].reshape(H, dh)
            v = qkv[:, 2 * D:].reshape(H, dh)
            if typ == "lin":
                Q, K = hn.phi(q), hn.phi(k)
                num = np.einsum("hi,hij->hj", Q, self.S[l])
                den = np.einsum("hi,hi->h", Q, self.Z[l]) + hn.EPS_DEN
                y = (num / den[:, None]).reshape(1, D)
                self.S[l] += np.einsum("hi,hj->hij", K, v)
                self.Z[l] += K
            else:
                Kc, Vc, n = self.kv[l]
                # le token courant entre d'abord dans son propre cache (diagonale causale)
                Kc[n], Vc[n] = k, v
                self.kv[l][2] = n + 1
                scores = self.scale * np.einsum("thd,hd->ht", Kc[:n + 1], q)    # (H, t+1)
                scores = scores - scores.max(-1, keepdims=True)
                w = np.exp(scores)
                w /= w.sum(-1, keepdims=True)
                y = np.einsum("ht,thd->hd", w, Vc[:n + 1]).reshape(1, D)
            x = x + y @ P[f"b{l}.proj.W"] + P[f"b{l}.proj.b"]
            h2, _, _ = hn.layer_norm_fw(x, P[f"b{l}.ln2.g"], P[f"b{l}.ln2.b"])
            a = h2 @ P[f"b{l}.fc1.W"] + P[f"b{l}.fc1.b"]
            x = x + hn.gelu(a) @ P[f"b{l}.fc2.W"] + P[f"b{l}.fc2.b"]
        xf, _, _ = hn.layer_norm_fw(x, P["lnf.g"], P["lnf.b"])
        self.t += 1
        return (xf @ P["tok_emb"].T)[0]

    def longueur_kv(self):
        tot = 0
        for l, t in enumerate(self.types):
            if t == "smax":
                tot += self.kv[l][2]
        return tot


def memoire(types, D, P, t_ctx):
    poids = sum(v.nbytes for v in P.values())
    etat = 0
    kv = 0
    for t in types:
        if t == "lin":
            etat += (D * D + D) * 4          # S (D×D) + Z (D)
        else:
            kv += 2 * t_ctx * D * 4          # K + V, float32
    return {"poids_octets": poids, "etat_lin_octets": etat, "kv_octets": kv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--types", required=True, help='ex: "lin,smax"')
    ap.add_argument("--data", default="data/corpus_fr.txt")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ids, chars = charger_corpus(args.data)
    P = {k: v for k, v in np.load(args.ckpt).items()}
    types = [t.strip() for t in args.types.split(",")]
    cfg = {"vocab_size": len(chars), "d_model": args.d_model, "n_head": args.n_head,
           "types": types, "block_size": 96}
    V = len(chars)

    print(f"[PROFILEUR] types={types} | ckpt={args.ckpt}")

    # ---------- 1. GARDE-FOU : équivalence incrémental ↔ parallèle ----------
    T_eq = 96
    logits_par, _, _ = hn.forward(P, cfg, ids[:T_eq][None])
    g = GenerationIncrementale(P, types, args.n_head)
    pire = 0.0
    for t in range(T_eq):
        lg = g.pas(ids[t])
        pire = max(pire, float(np.abs(lg - logits_par[0, t]).max()))
    print(f"garde-fou équivalence : écart max logits = {pire:.2e} "
          f"({'✔ < 2e-3' if pire < 2e-3 else '✘ TROP GRAND — mesures invalidées'})")
    if pire >= 2e-3:
        sys.exit(1)

    # ---------- 2. V_gen mesuré (dans la distribution) ----------
    vgen = {}
    for T0 in (16, 48, 96):
        g.reset()
        t0 = time.perf_counter()
        for i in range(T0):
            g.pas(ids[i])
        prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        tok = ids[T0 - 1]
        for _ in range(args.n_gen):
            lg = g.pas(tok)
            tok = int(np.argmax(lg))
        dt = time.perf_counter() - t0
        vgen[T0] = {"tok_s": round(args.n_gen / dt, 1),
                    "prefill_tok_s": round(T0 / prefill, 1),
                    "kv_a_mi_trajet_octets": memoire(types, args.d_model, P, T0 + args.n_gen // 2)["kv_octets"]}
        print(f"V_gen @T0={T0:>3} : {vgen[T0]['tok_s']:.1f} tok/s (prefill {vgen[T0]['prefill_tok_s']:.0f} tok/s)")

    # ---------- 3. contraste naïf (re-calcul de la fenêtre) ----------
    naive = None
    if "smax" in types:
        T0 = 48
        t0 = time.perf_counter()
        ctx = list(ids[:T0])
        for _ in range(60):
            lg, _, _ = hn.forward(P, cfg, np.array(ctx[-96:])[None])
            ctx.append(int(np.argmax(lg[0, -1])))
        dt = time.perf_counter() - t0
        naive = round(60 / dt, 1)
        print(f"V_gen naïve (re-calcul fenêtre complète) : {naive} tok/s")

    # ---------- 4. scaling mécanique (pos_emb neutralisée) ----------
    P2 = dict(P)
    P2["pos_emb"] = np.zeros_like(P["pos_emb"])
    scaling = {}
    for T in (256, 1024, 4096):
        g2 = GenerationIncrementale(P2, types, args.n_head, Tmax=T + 100)
        t0 = time.perf_counter()
        for i in range(T):
            g2.pas(ids[i % len(ids)])
        prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        tok = ids[(T - 1) % len(ids)]
        for _ in range(100):
            lg = g2.pas(tok)
            tok = int(np.argmax(lg))
        dt = time.perf_counter() - t0
        m = memoire(types, args.d_model, P2, T)
        scaling[T] = {"prefill_tok_s": round(T / prefill, 1),
                      "v_gen_tok_s": round(100 / dt, 1),
                      "kv_octets": m["kv_octets"]}
        print(f"scaling T={T:>5} : prefill {scaling[T]['prefill_tok_s']:.0f} tok/s | "
              f"V_gen {scaling[T]['v_gen_tok_s']:.1f} tok/s | KV {m['kv_octets']/1024:.0f} Ko")

    # ---------- 5. mémoire ----------
    mem = memoire(types, args.d_model, P, 96)
    print(f"mémoire : poids {mem['poids_octets']/1024:.0f} Ko | état linéaire {mem['etat_lin_octets']/1024:.0f} Ko | "
          f"KV@96 {mem['kv_octets']/1024:.0f} Ko | KV@4096 {memoire(types, args.d_model, P, 4096)['kv_octets']/1024:.0f} Ko")

    resultat = {"types": types, "ckpt": args.ckpt,
                "equivalence_max_diff": pire,
                "v_gen_dans_distribution": vgen,
                "v_gen_naive_fenetre": naive,
                "scaling_mecanisme": scaling,
                "memoire": mem,
                "memoire_kv_4096": memoire(types, args.d_model, P, 4096)}
    chemin = args.out or f"results/profile_inf_{args.types.replace(',', '-')}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultat, f, indent=1, ensure_ascii=False)
    print("Écrit :", chemin)


if __name__ == "__main__":
    main()
