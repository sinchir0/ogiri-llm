"""固定評価お題の切り出し。  python -m ogiri.benchmark build --src data/topics.jsonl --n 500"""
import argparse
import random

from .common import DATA, read_jsonl, topic_hash, write_jsonl


def build(src, n, seed=0):
    topics = sorted({r["topic"].strip() for r in read_jsonl(src)})
    random.Random(seed).shuffle(topics)
    bench, train = topics[:n], topics[n:]
    write_jsonl(DATA / "bench_topics.jsonl", [{"topic": t} for t in bench])
    write_jsonl(DATA / "train_topics.jsonl", [{"topic": t} for t in train])
    print(f"bench={len(bench)} train={len(train)}")


def bench_hashes():
    return {topic_hash(r["topic"], r.get("image")) for r in read_jsonl(DATA / "bench_topics.jsonl")}


def filter_train(rows):
    """学習データからベンチお題を除外する(リーク防止)。"""
    bad = bench_hashes()
    return [r for r in rows if topic_hash(r.get("topic"), r.get("image")) not in bad]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--src", default=str(DATA / "topics.jsonl"))
    p.add_argument("--n", type=int, default=500)
    a = p.parse_args()
    build(a.src, a.n)
