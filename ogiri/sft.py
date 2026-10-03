"""bf16 LoRA SFT (Qwen3.5, テキスト+画像混在)。  python -m ogiri.sft --out ckpt/sft

Qwen3.5はQLoRA(4bit)非推奨(量子化誤差が大きい)のため bf16 LoRA を使う。
data/sft.jsonl(テキストのみ) と data/sft_clot.jsonl(画像)を混合して学習する。
回答部分のみに損失をかけるため、collate_fnでプロンプト部分のlabelsを-100にマスクする。
"""
import argparse

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from .benchmark import filter_train
from .common import BASE_MODEL, DATA, chat, read_jsonl

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(data, data_clot):
    text_rows = [r for r in filter_train(read_jsonl(data)) if r.get("tier", "win") == "win"]
    img_rows = [r for r in filter_train(read_jsonl(data_clot)) if r.get("tier", "win") == "win"]
    rows = text_rows + img_rows
    assert rows, f"no SFT rows in {data} / {data_clot}"
    return rows


def make_collate_fn(processor):
    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(examples):
        full_ids, mm_types, prompt_lens, pixels, grids = [], [], [], [], []
        for ex in examples:
            img_path = ex.get("image")
            msgs = chat(ex.get("topic"), ex["answer"], image=img_path)
            imgs = [Image.open(img_path).convert("RGB")] if img_path else None
            full_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt_text = processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            enc_full = processor(text=[full_text], images=[imgs] if imgs else None, return_tensors="pt")
            enc_prompt = processor(text=[prompt_text], images=[imgs] if imgs else None, return_tensors="pt")
            full_ids.append(enc_full["input_ids"][0])
            mm_types.append(enc_full["mm_token_type_ids"][0])
            prompt_lens.append(enc_prompt["input_ids"].shape[1])
            if imgs:
                pixels.append(enc_full["pixel_values"])
                grids.append(enc_full["image_grid_thw"])

        maxlen = max(x.shape[0] for x in full_ids)
        n = len(examples)
        input_ids = torch.full((n, maxlen), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, maxlen), dtype=torch.long)
        mm_token_type_ids = torch.zeros((n, maxlen), dtype=torch.long)
        labels = torch.full((n, maxlen), -100, dtype=torch.long)
        for i, ids in enumerate(full_ids):
            L = ids.shape[0]
            input_ids[i, :L] = ids
            attention_mask[i, :L] = 1
            mm_token_type_ids[i, :L] = mm_types[i]
            labels[i, prompt_lens[i]:L] = ids[prompt_lens[i]:]

        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "mm_token_type_ids": mm_token_type_ids, "labels": labels}
        if pixels:
            batch["pixel_values"] = torch.cat(pixels, dim=0)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch

    return collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA / "sft.jsonl"))
    p.add_argument("--data_clot", default=str(DATA / "sft_clot.jsonl"))
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", default="ckpt/sft")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--limit", type=int, help="デバッグ用: 先頭N件のみ使う")
    a = p.parse_args()

    rows = load_rows(a.data, a.data_clot)[: a.limit]
    examples = [{"topic": r.get("topic"), "answer": r["answer"],
                 "image": str(DATA / r["image"]) if r.get("image") else None} for r in rows]
    ds = Dataset.from_list(examples)

    processor = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16)
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0.03, bf16=True, logging_steps=10,
        save_strategy="epoch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
    )
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=TARGET_MODULES)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=processor,
                    data_collator=make_collate_fn(processor), peft_config=lora)
    tr.train()
    tr.save_model(a.out)


if __name__ == "__main__":
    main()
