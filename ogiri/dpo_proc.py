"""手順SFT済みモデルの DPO (思考+回答 の全体を比較。テキスト/画像/画像+テキスト対応)。
python -m ogiri.dpo_proc --init ckpt/sft_proc --prefs data/prefs_proc.jsonl --out ckpt/dpo_proc

- 方策 = base + SFT LoRA (--init) をそのまま学習可能にして更新する。参照モデルは別に持たず、
  学習前に全ペアの参照 logp を1回だけ計算して保持する(= 初期の SFT モデルが参照)。
- 損失 = DPO(sigmoid) + alpha * NLL(chosen)。ペアが少ないので、選ばれた案への SFT 項で崩れを抑える (RPO 風)。
- お題(画像含む)単位で検証に回し、報酬の正解率(chosen の暗黙報酬 > rejected)を epoch ごとに出す。
入力: prefs の各行 {"topic","image","chosen","rejected","chosen_think","rejected_think"}
"""
import argparse
import random

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .common import BASE_MODEL, DATA, chat, read_jsonl, topic_hash
from .prompts import PROC_SYSTEM


def encode(processor, topic, image, think, answer):
    """(input_ids, mm_token_type_ids, pixel_values, image_grid_thw, prompt_len)。SFT の手順形式と同じ組み立て。"""
    path = str(DATA / image) if image else None
    imgs = [Image.open(path).convert("RGB")] if path else None
    msgs = chat(topic, None, image=path, system=PROC_SYSTEM)
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert prompt.endswith("<think>\n")
    full = prompt + f"{think}\n</think>\n\n{answer}<|im_end|>\n"
    ef = processor(text=[full], images=[imgs] if imgs else None, return_tensors="pt")
    ep = processor(text=[prompt], images=[imgs] if imgs else None, return_tensors="pt")
    return (ef["input_ids"][0], ef["mm_token_type_ids"][0], ef.get("pixel_values"), ef.get("image_grid_thw"),
            ep["input_ids"].shape[1])


def make_batch(processor, pair, device):
    """chosen / rejected の2系列を1バッチにする(同じお題・同じ画像)。"""
    enc = [encode(processor, pair["topic"], pair.get("image"), pair["chosen_think"], pair["chosen"]),
           encode(processor, pair["topic"], pair.get("image"), pair["rejected_think"], pair["rejected"])]
    L = max(e[0].shape[0] for e in enc)
    pad = processor.tokenizer.pad_token_id
    ids = torch.full((2, L), pad, dtype=torch.long)
    att = torch.zeros((2, L), dtype=torch.long)
    mm = torch.zeros((2, L), dtype=torch.long)
    for i, e in enumerate(enc):
        n = e[0].shape[0]
        ids[i, :n], att[i, :n], mm[i, :n] = e[0], 1, e[1]
    batch = {"input_ids": ids, "attention_mask": att, "mm_token_type_ids": mm}
    if enc[0][2] is not None:
        batch["pixel_values"] = torch.cat([e[2] for e in enc], dim=0)
        batch["image_grid_thw"] = torch.cat([e[3] for e in enc], dim=0)
    plen = [e[4] for e in enc]
    lens = [e[0].shape[0] for e in enc]
    return {k: v.to(device) for k, v in batch.items()}, plen, lens


def completion_logps(model, batch, plen, lens):
    """各系列の 完了部分(思考+回答)の logp 合計と トークン数。"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**batch).logits
    out = []
    for i in range(2):
        lo, hi = plen[i] - 1, lens[i] - 1  # 位置 t のlogitsは t+1 のトークンを予測
        lp = F.log_softmax(logits[i, lo:hi].float(), dim=-1)
        tgt = batch["input_ids"][i, plen[i]:lens[i]]
        out.append(lp.gather(-1, tgt[:, None]).squeeze(-1).sum())
    return out[0], out[1], lens[0] - plen[0]


def dpo_terms(pc, pr, rc, rr, beta):
    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin), margin


@torch.no_grad()
def reference_logps(model, processor, pairs, device):
    model.eval()
    ref = []
    for p in pairs:
        batch, plen, lens = make_batch(processor, p, device)
        c, r, _ = completion_logps(model, batch, plen, lens)
        ref.append((c.item(), r.item()))
    return ref


@torch.no_grad()
def evaluate(model, processor, pairs, ref, beta, device):
    model.eval()
    margins = []
    for p, (rc, rr) in zip(pairs, ref):
        batch, plen, lens = make_batch(processor, p, device)
        c, r, _ = completion_logps(model, batch, plen, lens)
        margins.append((beta * ((c.item() - rc) - (r.item() - rr))))
    acc = sum(m > 0 for m in margins) / max(len(margins), 1)
    return acc, sum(margins) / max(len(margins), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--init", required=True, help="手順SFTのLoRA (例 ckpt/sft_proc)")
    ap.add_argument("--prefs", default=str(DATA / "prefs_proc.jsonl"))
    ap.add_argument("--out", default="ckpt/dpo_proc")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--alpha", type=float, default=0.1, help="chosen のNLL(トークン平均)の重み")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8, help="1ステップあたりのペア数")
    ap.add_argument("--val_frac", type=float, default=0.15, help="検証に回すお題の割合")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="デバッグ用: 先頭Nペアのみ")
    a = ap.parse_args()

    rows = [r for r in read_jsonl(a.prefs) if r.get("chosen_think") and r.get("rejected_think")]
    rows = rows[: a.limit]
    assert rows, "思考つきの選好ペアがありません(arena で投票してください)"
    keys = sorted({topic_hash(r["topic"], r.get("image")) for r in rows})
    rnd = random.Random(a.seed)
    rnd.shuffle(keys)
    val_keys = set(keys[: max(1, int(len(keys) * a.val_frac))]) if len(keys) >= 8 else set()
    train = [r for r in rows if topic_hash(r["topic"], r.get("image")) not in val_keys]
    val = [r for r in rows if topic_hash(r["topic"], r.get("image")) in val_keys]
    print(f"pairs: train {len(train)} / val {len(val)} (お題 {len(keys)}, 検証お題 {len(val_keys)})", flush=True)

    device = "cuda"
    processor = AutoProcessor.from_pretrained(a.base)
    base = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    ref_tr = reference_logps(model, processor, train, device)
    ref_va = reference_logps(model, processor, val, device) if val else []
    if val:
        acc, mg = evaluate(model, processor, val, ref_va, a.beta, device)
        print(f"[epoch 0] val acc {acc:.0%} margin {mg:+.4f} (初期。acc は 0% 付近=差なしが正常)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total_steps = max(1, (len(train) * a.epochs) // a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * max(0.1, 1 - s / total_steps))
    step = 0
    for ep in range(1, a.epochs + 1):
        order = list(range(len(train)))
        rnd.shuffle(order)
        model.train()
        run_loss = run_acc = 0.0
        for k, idx in enumerate(order, 1):
            p, (rc, rr) = train[idx], ref_tr[idx]
            batch, plen, lens = make_batch(processor, p, device)
            pc, pr, nc = completion_logps(model, batch, plen, lens)
            dpo, margin = dpo_terms(pc, pr, rc, rr, a.beta)
            loss = (dpo + a.alpha * (-pc / nc)) / a.accum
            loss.backward()
            run_loss += dpo.item()
            run_acc += float(margin.item() > 0)
            if k % a.accum == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                step += 1
        msg = f"[epoch {ep}] train dpo_loss {run_loss / len(order):.4f} acc {run_acc / len(order):.0%}"
        if val:
            acc, mg = evaluate(model, processor, val, ref_va, a.beta, device)
            msg += f" | val acc {acc:.0%} margin {mg:+.4f}"
        print(msg, flush=True)
    model.save_pretrained(a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
