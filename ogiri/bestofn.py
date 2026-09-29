"""推論時 Best-of-N:  Generator が N 案 → Judge で上位 K 案。
python -m ogiri.bestofn "お題文" --model Qwen/Qwen3-8B --lora ckpt/dpo -n 32 -k 3
"""
import argparse

from .generate import generate
from .judge import score_all


def best_of_n(topic, model, lora=None, n=32, k=3, judge=None):
    cands = generate([topic], model, lora, n)
    scored = score_all(cands, judge or model)[0]["scored"]
    return sorted(scored, key=lambda x: -x["total"])[:k]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("topic")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora")
    p.add_argument("-n", type=int, default=32)
    p.add_argument("-k", type=int, default=3)
    a = p.parse_args()
    for x in best_of_n(a.topic, a.model, a.lora, a.n, a.k):
        print(f"{x['total']:.2f}  {x['text']}")
