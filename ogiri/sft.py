"""QLoRA SFT。  python -m ogiri.sft --out ckpt/sft"""
import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .benchmark import filter_train
from .common import BASE_MODEL, DATA, chat, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA / "sft.jsonl"))
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", default="ckpt/sft")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()

    rows = [r for r in filter_train(read_jsonl(a.data)) if r.get("tier", "win") == "win"]
    assert rows, f"no SFT rows in {a.data}"
    ds = Dataset.from_list([{"messages": chat(r["topic"], r["answer"])} for r in rows])

    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=10,
        save_strategy="epoch", max_length=512, assistant_only_loss=False,
        model_init_kwargs=dict(
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        ),
    )
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules="all-linear")
    tr = SFTTrainer(model=a.base, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    tr.train()
    tr.save_model(a.out)


if __name__ == "__main__":
    main()
