"""大喜利Judge: 5軸採点 → 選好ペア作成。  python -m ogiri.judge --cands data/cands.jsonl

採点は1-10の整数を出力させ、その確率分布の期待値を使う(サンプリングより低ノイズ)。
"""
import argparse
import math
import random

from .common import DATA, JUDGE_MODEL, read_jsonl, write_jsonl

AXES = {
    "funny": "純粋に笑えるか",
    "unexpected": "意外性・発想の飛躍があるか",
    "relevance": "お題に的確に絡んでいるか(意味不明でないか)",
    "original": "既視感のなさ(ありがちな定番回答でないか)",
    "brevity": "短く切れ味があるか",
}
WEIGHTS = {"funny": 0.4, "unexpected": 0.2, "relevance": 0.2, "original": 0.15, "brevity": 0.05}


def _prompt(topic, ans, axis):
    return [
        {"role": "system", "content": "あなたは厳しい大喜利の審査員です。1〜9の整数のみを出力します。5は平凡、9は優勝級。"},
        {"role": "user", "content": f"お題: {topic}\n回答: {ans}\n\n評価軸: {AXES[axis]}\n点数(1-9):"},
    ]


def score_all(rows, model=JUDGE_MODEL):
    """rows: [{topic, candidates}] -> [{topic, scored:[{text, scores{axis}, total}]}]"""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=1024, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    jobs = [(i, j, ax) for i, r in enumerate(rows) for j in range(len(r["candidates"])) for ax in AXES]
    msgs = [_prompt(rows[i]["topic"], rows[i]["candidates"][j], ax) for i, j, ax in jobs]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    tmp = {}
    for (i, j, ax), o in zip(jobs, outs):
        lp = o.outputs[0].logprobs[0]
        num = den = 0.0
        for tid, l in lp.items():
            t = (l.decoded_token or "").strip()
            if t.isdigit() and 1 <= int(t) <= 9:
                p = math.exp(l.logprob)
                num += int(t) * p
                den += p
        tmp[(i, j, ax)] = num / den if den else 5.0
    res = []
    for i, r in enumerate(rows):
        sc = []
        for j, c in enumerate(r["candidates"]):
            s = {ax: tmp[(i, j, ax)] for ax in AXES}
            sc.append({"text": c, "scores": s, "total": sum(WEIGHTS[a] * v for a, v in s.items())})
        res.append({"topic": r["topic"], "scored": sc})
    return res


def make_pairs(scored, top_k=4, bot_k=4, min_gap=1.0, min_relevance=4.0, seed=0):
    """上位 vs 下位でペア化。意味不明(relevance低)な上位は除外して「うまく外す」より「壊れた」を避ける。"""
    rng = random.Random(seed)
    pairs = []
    for r in scored:
        s = sorted(r["scored"], key=lambda x: -x["total"])
        top = [x for x in s if x["scores"]["relevance"] >= min_relevance][:top_k]
        bot = s[-bot_k:]
        for c in top:
            for rj in bot:
                if c["total"] - rj["total"] >= min_gap:
                    pairs.append({"topic": r["topic"], "chosen": c["text"], "rejected": rj["text"], "source": "judge"})
    rng.shuffle(pairs)
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cands", default=str(DATA / "cands.jsonl"))
    p.add_argument("--scored", default=str(DATA / "scored.jsonl"))
    p.add_argument("--prefs", default=str(DATA / "prefs.jsonl"))
    p.add_argument("--model", default=JUDGE_MODEL)
    a = p.parse_args()
    scored = score_all(read_jsonl(a.cands), a.model)
    write_jsonl(a.scored, scored)
    pairs = make_pairs(scored)
    write_jsonl(a.prefs, pairs, mode="a")  # 人間ペアと同居させるため追記
    print(f"pairs={len(pairs)}")
