"""DPO/SimPO 学習。  python -m ogiri.dpo --init ckpt/sft --out ckpt/dpo

--init は SFT の LoRA アダプタ。マージして base とし、新しい LoRA を DPO で学習する。
"""
import argparse

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .benchmark import filter_train
from .common import BASE_MODEL, DATA, chat, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--init", help="SFT LoRA dir")
    p.add_argument("--prefs", default=str(DATA / "prefs.jsonl"))
    p.add_argument("--out", default="ckpt/dpo")
    p.add_argument("--loss", default="sigmoid", help="sigmoid(DPO)|ipo|simpo等 TRL loss_type")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--human_weight", type=int, default=3, help="人間ペアの重複回数(Judgeより信頼)")
    a = p.parse_args()

    rows = filter_train(read_jsonl(a.prefs))
    data = []
    for r in rows:
        k = a.human_weight if r.get("source") == "human" else 1
        ex = {"prompt": chat(r["topic"]), "chosen": [{"role": "assistant", "content": r["chosen"]}],
              "rejected": [{"role": "assistant", "content": r["rejected"]}]}
        data += [ex] * k
    assert data, "no preference pairs"

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    if a.init:
        model = PeftModel.from_pretrained(model, a.init).merge_and_unload()
    cfg = DPOConfig(output_dir=a.out, beta=a.beta, learning_rate=a.lr, loss_type=a.loss,
                    per_device_train_batch_size=4, gradient_accumulation_steps=8, num_train_epochs=1,
                    bf16=True, logging_steps=10, max_length=512, gradient_checkpointing=True)
    lora = LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM", target_modules="all-linear")
    tr = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(data),
                    processing_class=tok, peft_config=lora)
    tr.train()
    tr.save_model(a.out)


if __name__ == "__main__":
    main()
