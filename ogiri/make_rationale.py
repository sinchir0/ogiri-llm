"""手順SFT用の教師データ作成: 人間に高評価された回答から、思考(5行)を逆算してベースモデルに書かせる。
python -m ogiri.make_rationale --n_text 10 --n_img 10 --out data/sft_proc_pilot.jsonl   # 品質確認用
python -m ogiri.make_rationale --n_text 4407 --n_img 3600 --out data/sft_proc.jsonl       # 本番
出力: {"topic","image","answer","think"}  think は "1. ありがち: ...\\n...\\n5. 具体化: ..." の5行
"""
import argparse
import random
import re

from .benchmark import filter_train
from .common import BASE_MODEL, DATA, read_jsonl, write_jsonl
from .prompts import RATIONALE_SYSTEM

LABELS = ["ありがち", "ヒンジ", "遠い世界", "衝突と理屈", "具体化"]
PREFILL = "1. ありがち:"
BAD = re.compile(r"[Ѐ-ӿ฀-๿가-힯؀-ۿ]")


def pick_rows(n_text, n_img, seed=0):
    rnd = random.Random(seed)
    text = [r for r in filter_train(read_jsonl(DATA / "sft.jsonl")) if r.get("tier", "win") == "win"]
    img = [r for r in filter_train(read_jsonl(DATA / "sft_clot.jsonl")) if r.get("tier", "win") == "win"]
    img = [r for r in img if len(r["answer"]) <= 60]  # 長すぎる回答は避ける
    rnd.shuffle(text), rnd.shuffle(img)
    return text[:n_text] + img[:n_img]


def parse(out, answer):
    """5行形式として妥当なら think 文字列を返し、そうでなければ None。"""
    lines = [l.strip() for l in (PREFILL + out).strip().split("\n") if l.strip()][:5]
    if len(lines) != 5:
        return None
    for i, (l, lab) in enumerate(zip(lines, LABELS), 1):
        if not l.startswith(f"{i}. {lab}:") or len(l) > 70 or BAD.search(l):
            return None
    think = "\n".join(lines)
    if len(answer) >= 8 and answer in think:  # 回答の丸写しは捨てる
        return None
    return think


def main():
    from PIL import Image
    from vllm import LLM, SamplingParams

    p = argparse.ArgumentParser()
    p.add_argument("--n_text", type=int, default=10)
    p.add_argument("--n_img", type=int, default=10)
    p.add_argument("--out", default=str(DATA / "sft_proc_pilot.jsonl"))
    p.add_argument("--temperature", type=float, default=0.5)
    a = p.parse_args()

    rows = pick_rows(a.n_text, a.n_img)
    llm = LLM(model=BASE_MODEL, max_model_len=4096, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 1})
    tok = llm.get_tokenizer()
    reqs = []
    for r in rows:
        img = Image.open(DATA / r["image"]).convert("RGB") if r.get("image") else None
        q = f"お題: {r['topic']}" if r.get("topic") else "お題: (画像)"
        uc = [{"type": "image"}, {"type": "text", "text": f"{q}\n回答: {r['answer']}"}] if img else f"{q}\n回答: {r['answer']}"
        prompt = tok.apply_chat_template([{"role": "system", "content": RATIONALE_SYSTEM}, {"role": "user", "content": uc}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) + PREFILL
        reqs.append({"prompt": prompt, "multi_modal_data": {"image": img}} if img else prompt)
    outs = llm.generate(reqs, SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=260))
    res, bad = [], 0
    for r, o in zip(rows, outs):
        think = parse(o.outputs[0].text, r["answer"])
        if think is None:
            bad += 1
            continue
        res.append({"topic": r.get("topic"), "image": r.get("image"), "answer": r["answer"], "think": think})
    print(f"有効 {len(res)}/{len(rows)} (不正形式で除外 {bad})", flush=True)
    write_jsonl(a.out, res)


if __name__ == "__main__":
    main()
