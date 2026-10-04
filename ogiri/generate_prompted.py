"""手順プロンプト条件の生成(vLLMを1回ロードして全条件)。テキスト/画像/画像+テキストのお題に対応。
python -m ogiri.generate_prompted --lora ckpt/sft_vlm --n 4
python -m ogiri.generate_prompted --items data/image_topics.jsonl --out data/arena_pool_image.jsonl
入力: {"topic": str|null, "image": "images/..."|null} の JSONL (topic のみのファイルも可)
出力: {"topic","image","model","style","answer"} JSONL (arena --per-model 用プール。model=条件名)
  A: sft + 従来SYSTEM(思考OFF) / B: base + FULL(思考ON) / C: sft + SHORT(思考OFF) / D: base + SHORT_COT(思考ON)
"""
import argparse
import re

from .common import BASE_MODEL, DATA, SYSTEM, read_jsonl, write_jsonl
from .prompts import FULL, PROC_SYSTEM, SHORT, SHORT_COT

CONDS = {  # 条件名: (LoRAを使うか, system, 思考ON/OFF)
    "A_sft_plain": (True, SYSTEM, False),
    "B_base_full": (False, FULL, True),
    "C_sft_short": (True, SHORT, False),
    "D_base_cot": (False, SHORT_COT, True),
    "E_jp_full": (False, FULL, True),    # 手順全文 + 日本語・短い思考(予算 --jp_think)
    "F_jp_cot": (False, SHORT_COT, True),  # 短いCoT + 日本語・短い思考
    "G_twostage": (False, SHORT, False),  # 思考なし2段階: 発想メモ -> 回答
    "H_proc": (True, PROC_SYSTEM, True),  # 手順SFT済みLoRA(--lora ckpt/sft_proc)。思考は自前で短い。思考も保存する
}
PROC_CONDS = {"H_proc"}
JP_CONDS = {"E_jp_full", "F_jp_cot"}
JP_HINT = "\n\n(内部の思考は日本語で、箇条書き5行以内。候補は最大3案まで。英語で書かない。)"
JP_PREFILL = "日本語で手短にメモします。\n1. ありがちな答え:"  # <think>\n の直後に差し込み、日本語・短文の思考に誘導する
MEMO_SYSTEM = (
    "あなたは大喜利の発想係です。お題(や画像)を見て、次を日本語で各1行、合計4行だけメモしてください。"
    "ボケの回答そのものは書かないでください。\n"
    "ヒンジ: お題の中で意味をずらせる語・物・状況\n"
    "遠い世界: 意外な別領域(会社・役所・ゲーム・学校・医療など)\n"
    "衝突: 二つの世界の対立(例: 真面目↔くだらない)\n"
    "変な理屈: 文字通り解釈/役割逆転/誇張/偽の因果 など、二つをつなぐ一本の理屈"
)
THINK_HINT = "\n\n(内部の思考は簡潔に。候補は最大5案まで。長く列挙しない。)"
FORCE = "\n\nThinking budget reached. I must stop and give the final answer now.\n</think>\n\n"
BAD = re.compile(r"[Ѐ-ӿ฀-๿가-힯؀-ۿ]")


def clean(text, thinking, topic=""):
    if thinking:
        if "</think>" not in text:  # 思考が打ち切られた出力は捨てる
            return None
        text = text.rsplit("</think>", 1)[1]
    text = text.strip().split("\n")[0].strip()
    if len(text) > 2 and (text[0], text[-1]) in {("「", "」"), ('"', '"'), ("『", "』")}:
        text = text[1:-1].strip()
    if topic and (text in topic or topic.rstrip("？?") in text):  # お題の復唱は捨てる
        return None
    return text if 2 <= len(text) <= 80 and not BAD.search(text) else None


def user_content(topic, image):
    """テキストのみは従来どおり文字列、画像ありはblock形式(画像は multi_modal_data で渡す)。"""
    text = f"お題: {topic}" if topic else "この画像で大喜利に回答してください。"
    return [{"type": "image"}, {"type": "text", "text": text}] if image else text


def collect(name, items, texts, thinking, keep_think=False):
    rows, total, good = [], 0, 0
    for it, ts in zip(items, texts):
        seen = set()
        for c in ts:
            total += 1
            a = clean(c, thinking, it["topic"])
            if a and a not in seen:
                seen.add(a)
                good += 1
                row = {"topic": it["topic"], "image": it["image"], "model": name, "style": "none", "answer": a}
                if keep_think:  # 思考(DPOで 思考+回答 を比較するため保存)。画面には出さない
                    row["think"] = c.split("</think>", 1)[0].strip()
                rows.append(row)
    print(f"[{name}] 有効 {good}/{total}", flush=True)
    return rows


def two_stage(llm, tok, items, imgs, n, temperature, top_p, req):
    """思考なし2段階: (1) 発想メモをn通り (2) 各メモを使って回答を1つ。"""
    from vllm import SamplingParams

    def render(system, it, extra=""):
        uc = user_content(it["topic"], it["image"])
        if extra:
            uc = (uc + [{"type": "text", "text": extra}]) if isinstance(uc, list) else uc + extra
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": uc}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    p1 = [render(MEMO_SYSTEM, it) for it in items]
    o1 = llm.generate([req(p, im) for p, im in zip(p1, imgs)],
                      SamplingParams(n=n, temperature=0.9, top_p=top_p, max_tokens=200))
    memos = [[c.text.strip() for c in o.outputs] for o in o1]
    p2, idx = [], []
    for i, (it, ms) in enumerate(zip(items, memos)):
        for j, m in enumerate(ms):
            p2.append(req(render(SHORT, it, f"\n\n【発想メモ】\n{m}\n\nこのメモの理屈を使い、メモの文言は書かず、回答を1つだけ短く出力してください。"), imgs[i]))
            idx.append(i)
    o2 = llm.generate(p2, SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=80))
    texts = [[] for _ in items]
    for i, o in zip(idx, o2):
        texts[i].append(o.outputs[0].text)
    return texts


def run(items, lora, n, temperature, top_p, conds, max_think, jp_think=512, proc_think=500):
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64, max_model_len=max_think + 3072,
              gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 1})
    tok = llm.get_tokenizer()
    imgs = [Image.open(DATA / it["image"]).convert("RGB") if it["image"] else None for it in items]

    def req(prompt, img):  # 画像があるときだけ multi_modal_data を付ける
        return {"prompt": prompt, "multi_modal_data": {"image": img}} if img is not None else prompt

    rows = []
    for name in conds:
        use_lora, system, thinking = CONDS[name]
        lora_req = LoRARequest("l", 1, lora) if use_lora else None
        jp = name in JP_CONDS
        proc = name in PROC_CONDS
        budget = jp_think if jp else proc_think if proc else max_think
        sysmsg = system + (JP_HINT if jp else THINK_HINT if thinking and not proc else "")
        prompts = [tok.apply_chat_template([{"role": "system", "content": sysmsg},
                                            {"role": "user", "content": user_content(it["topic"], it["image"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for it in items]
        if jp:
            prompts = [pr + JP_PREFILL for pr in prompts]
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=budget if thinking else 80)
        if name == "G_twostage":
            texts = two_stage(llm, tok, items, imgs, n, temperature, top_p, req)
            rows += collect(name, items, texts, False)
            continue
        outs = llm.generate([req(p, im) for p, im in zip(prompts, imgs)], sp, lora_request=lora_req)
        texts = [[c.text for c in o.outputs] for o in outs]
        if thinking and not proc:  # 予算内に思考が終わらなかったものは </think> を強制挿入して回答させる
            idx, forced = [], []
            for i, (pr, ts) in enumerate(zip(prompts, texts)):
                for j, t in enumerate(ts):
                    if "</think>" not in t:
                        idx.append((i, j))
                        forced.append(req(pr + t + FORCE, imgs[i]))
            print(f"[{name}] 思考が予算超過: {len(idx)}/{len(items) * n} -> 強制終了して回答生成", flush=True)
            if forced:
                fo = llm.generate(forced, SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=80), lora_request=lora_req)
                for (i, j), o in zip(idx, fo):
                    texts[i][j] = "</think>" + o.outputs[0].text
        rows += collect(name, items, texts, thinking, keep_think=proc)  # procは思考が終わらなかった出力を捨てる
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default="ckpt/sft_vlm")
    p.add_argument("--items", "--topics", dest="items", default=str(DATA / "sample_topics.jsonl"))
    p.add_argument("--out", default=str(DATA / "arena_pool_prompted.jsonl"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--conds", nargs="+", default=list(CONDS), choices=list(CONDS))
    p.add_argument("--max_think", type=int, default=2048)
    p.add_argument("--proc_think", type=int, default=500, help="H_proc条件の最大思考トークン")
    p.add_argument("--jp_think", type=int, default=512, help="E/F条件の思考予算(トークン)")
    a = p.parse_args()
    items = [{"topic": r.get("topic") or None, "image": r.get("image") or None} for r in read_jsonl(a.items)][: a.limit]
    write_jsonl(a.out, run(items, a.lora, a.n, a.temperature, a.top_p, a.conds, a.max_think, a.jp_think, a.proc_think))
