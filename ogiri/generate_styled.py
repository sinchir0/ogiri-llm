"""方向性(スタイル)を指定して多様に生成し、arena_pool.jsonl を作る。
python -m ogiri.generate_styled --lora ckpt/sft --n_per_style 6

同じ LLM を1回だけロードし、base(LoRAなし)と sft(LoRAあり)の両方を生成する。
出力: {"topic","model","style","answer"} の JSONL
"""
import argparse
import re

from .common import BASE_MODEL, DATA, SYSTEM, read_jsonl, write_jsonl

STYLES = {
    "王道": "誰もが納得する、切れ味のある王道の答え",
    "誇張": "極端に誇張した答え",
    "言葉遊び": "駄洒落や言葉遊びを使った答え",
    "視点変更": "意外な人物・動物・物の視点から見た答え",
    "不条理": "不条理でシュールな答え",
    "具体的": "固有名詞や具体的な細部を入れた、短い答え",
}
BAD_CHARS = re.compile(r"[Ѐ-ӿ฀-๿가-힯؀-ۿ]")  # キリル/タイ/ハングル/アラビア


def ok(ans, topic):
    return 3 <= len(ans) <= 60 and not BAD_CHARS.search(ans) and ans not in topic


def run(topics, model, lora, n_per_style, temperature, top_p, min_p):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model, enable_lora=True, max_lora_rank=64, max_model_len=1024, gpu_memory_utilization=0.9)
    sp = SamplingParams(n=n_per_style, temperature=temperature, top_p=top_p, min_p=min_p, max_tokens=80)
    jobs = [(t, s) for t in topics for s in STYLES]
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"お題: {t}\n方向性: {STYLES[s]}"}] for t, s in jobs]
    pool = []
    for name, lr in [("base", None), ("sft", LoRARequest("l", 1, lora))]:
        outs = llm.chat(msgs, sp, lora_request=lr, chat_template_kwargs={"enable_thinking": False})
        for (t, s), o in zip(jobs, outs):
            seen = set()
            for c in o.outputs:
                a = c.text.strip().split("\n")[0].strip()
                if a and a not in seen and ok(a, t):
                    seen.add(a)
                    pool.append({"topic": t, "model": name, "style": s, "answer": a})
    return pool


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="ckpt/sft")
    p.add_argument("--topics", default=str(DATA / "topics.jsonl"))
    p.add_argument("--out", default=str(DATA / "arena_pool.jsonl"))
    p.add_argument("--n_per_style", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--min_p", type=float, default=0.05)
    a = p.parse_args()
    topics = [r["topic"] for r in read_jsonl(a.topics)]
    pool = run(topics, a.model, a.lora, a.n_per_style, a.temperature, a.top_p, a.min_p)
    write_jsonl(a.out, pool)
    print(f"pool={len(pool)}")
