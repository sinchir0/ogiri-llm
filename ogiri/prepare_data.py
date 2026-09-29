"""公開データ iammytoo/japanese-humor-evaluation-v2 から SFT/選好データを作る。
python -m ogiri.prepare_data

- テキストお題のみ(画像お題は odai が空なので除外)
- SFT: score >= --sft_min の回答
- 選好: 同一お題内で score >= --hi と score <= --lo を組み合わせ(該当お題のみ)
- 私(自作)の評価お題と完全一致するお題は除外(リーク防止)
※元データはNHK番組由来で権利未確認。研究目的に限ること。
"""
import argparse
import collections
import random

from datasets import load_dataset

from .common import DATA, read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_min", type=float, default=3.5)
    p.add_argument("--hi", type=float, default=3.5)
    p.add_argument("--lo", type=float, default=2.0)
    a = p.parse_args()

    ds = load_dataset("iammytoo/japanese-humor-evaluation-v2")
    rows = [dict(r) for s in ds for r in ds[s].remove_columns("image")]
    rows = [r for r in rows if r["odai_type"] == "text" and r["odai"].strip() and r["response"].strip()]
    mine = {r["topic"] for r in read_jsonl(DATA / "topics.jsonl")}
    rows = [r for r in rows if r["odai"].strip() not in mine]

    sft = [{"topic": r["odai"].strip(), "answer": r["response"].strip(), "score": r["score"], "tier": "win"}
           for r in rows if r["score"] >= a.sft_min]
    write_jsonl(DATA / "sft.jsonl", sft)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["odai"].strip()].append(r)
    prefs = []
    for t, rs in by.items():
        hi = [r for r in rs if r["score"] >= a.hi]
        lo = [r for r in rs if r["score"] <= a.lo]
        for h in hi:
            for l in random.Random(0).sample(lo, min(2, len(lo))):
                prefs.append({"topic": t, "chosen": h["response"].strip(), "rejected": l["response"].strip(),
                              "source": "public"})
    write_jsonl(DATA / "prefs_public.jsonl", prefs)
    print(f"sft={len(sft)} topics={len({r['topic'] for r in sft})} prefs_public={len(prefs)}")


if __name__ == "__main__":
    main()
