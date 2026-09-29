"""各お題にN案生成(vLLM)。  python -m ogiri.generate --model Qwen/Qwen3-8B --lora ckpt/sft --n 64"""
import argparse

from .common import BASE_MODEL, DATA, SYSTEM, read_jsonl, write_jsonl


def generate(topics, model, lora=None, n=64, temperature=1.1, top_p=0.95, max_tokens=80):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model, enable_lora=bool(lora), max_model_len=1024, gpu_memory_utilization=0.9)
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"お題: {t}"}] for t in topics]
    outs = llm.chat(msgs, sp, lora_request=LoRARequest("l", 1, lora) if lora else None,
                    chat_template_kwargs={"enable_thinking": False})
    res = []
    for t, o in zip(topics, outs):
        cands = list(dict.fromkeys(c.text.strip().split("\n")[0] for c in o.outputs if c.text.strip()))
        res.append({"topic": t, "candidates": cands})
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora")
    p.add_argument("--topics", default=str(DATA / "train_topics.jsonl"))
    p.add_argument("--out", default=str(DATA / "cands.jsonl"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    topics = [r["topic"] for r in read_jsonl(a.topics)][: a.limit]
    write_jsonl(a.out, generate(topics, a.model, a.lora, a.n))
