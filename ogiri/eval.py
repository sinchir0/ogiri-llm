"""対戦ログから Bradley–Terry 強さ + Elo を推定。  python -m ogiri.eval [--battles data/battles.jsonl]
引き分けは 0.5勝ずつ。ブートストラップで95%区間も出す。
"""
import argparse
import random
from collections import defaultdict

import numpy as np

from .common import DATA, read_jsonl


def bradley_terry(battles, iters=200):
    models = sorted({b["a_model"] for b in battles} | {b["b_model"] for b in battles})
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    w = np.zeros((n, n))
    for b in battles:
        i, j = idx[b["a_model"]], idx[b["b_model"]]
        if b["winner"] == "a":
            w[i, j] += 1
        elif b["winner"] == "b":
            w[j, i] += 1
        else:
            w[i, j] += 0.5
            w[j, i] += 0.5
    p = np.ones(n)
    for _ in range(iters):  # MM algorithm
        for i in range(n):
            num = w[i].sum()
            den = sum((w[i, j] + w[j, i]) / (p[i] + p[j]) for j in range(n) if j != i)
            p[i] = num / den if den > 0 and num > 0 else 1e-3
        p /= np.exp(np.log(p).mean())
    elo = 400 * np.log10(p) + 1000
    return dict(zip(models, elo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", default=str(DATA / "battles.jsonl"))
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    bs = read_jsonl(a.battles)
    base = bradley_terry(bs)
    boots = defaultdict(list)
    for s in range(a.boot):
        r = random.Random(s)
        for m, v in bradley_terry([r.choice(bs) for _ in bs], 50).items():
            boots[m].append(v)
    for m, v in sorted(base.items(), key=lambda x: -x[1]):
        lo, hi = np.percentile(boots[m], [2.5, 97.5]) if boots[m] else (v, v)
        print(f"{m:30s} {v:7.1f}  [{lo:.0f}, {hi:.0f}]")


if __name__ == "__main__":
    main()
