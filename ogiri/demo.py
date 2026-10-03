"""学習済みLoRAの生成デモ(テキストのみ/画像のみ/画像+テキストの3パターン)。
起動: python -m ogiri.demo --ckpt ckpt/trial100 --port 17081
"""
import argparse
import io

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from .common import BASE_MODEL, chat

app = FastAPI()
STATE = {}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>大喜利 生成デモ</title><style>
body{font:16px system-ui;max-width:720px;margin:1.5rem auto;padding:0 1rem}
section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
h2{margin:.2rem 0 1rem}
textarea,input[type=text]{width:100%;box-sizing:border-box;font-size:1rem;padding:.5rem}
button{padding:.6rem 1.2rem;margin-top:.5rem;cursor:pointer}
.out{margin-top:.8rem;padding:.6rem;background:#f6f6f6;border-radius:6px;white-space:pre-wrap;min-height:1.5em}
.loading{color:#888}
img.preview{max-width:100%;max-height:220px;display:block;margin-top:.5rem}
</style>
<h1>大喜利 生成デモ</h1>

<section>
<h2>① テキストのみ</h2>
<input type=text id=t1 placeholder="お題を入力" value="こんなコンビニは嫌だ。どんなコンビニ？">
<button onclick="run(1)">生成</button>
<div class=out id=o1></div>
</section>

<section>
<h2>② 画像のみ</h2>
<input type=file id=f2 accept="image/*" onchange="preview(2)"><img class=preview id=p2>
<button onclick="run(2)">生成</button>
<div class=out id=o2></div>
</section>

<section>
<h2>③ 画像 + テキスト</h2>
<input type=file id=f3 accept="image/*" onchange="preview(3)"><img class=preview id=p3>
<input type=text id=t3 placeholder="お題を入力(任意)" style="margin-top:.5rem">
<button onclick="run(3)">生成</button>
<div class=out id=o3></div>
</section>

<script>
function preview(n){
  const f = document.getElementById('f'+n).files[0];
  if(!f) return;
  document.getElementById('p'+n).src = URL.createObjectURL(f);
}
async function run(n){
  const out = document.getElementById('o'+n);
  out.textContent = '生成中...'; out.classList.add('loading');
  const fd = new FormData();
  const t = document.getElementById('t'+n);
  if(t && t.value.trim()) fd.append('topic', t.value.trim());
  const f = document.getElementById('f'+n);
  if(f && f.files[0]) fd.append('image', f.files[0]);
  try{
    const r = await fetch('/api/generate', {method:'POST', body: fd});
    const j = await r.json();
    out.textContent = j.answer ?? ('エラー: ' + j.error);
  }catch(e){ out.textContent = 'エラー: ' + e; }
  out.classList.remove('loading');
}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/generate")
async def generate(topic: str | None = Form(None), image: UploadFile | None = File(None)):
    try:
        img = None
        if image is not None and image.filename:
            img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        processor, model = STATE["processor"], STATE["model"]
        msgs = chat(topic, image="x" if img else None)  # "x"はプレースホルダ(実画像はimagesで渡す)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = processor(text=[text], images=[[img]] if img else None, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False)
        answer = processor.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--ckpt", default="ckpt/trial100")
    ap.add_argument("--port", type=int, default=17081)
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(a.base)
    base = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, a.ckpt).cuda()
    model.eval()
    STATE["processor"], STATE["model"] = processor, model

    uvicorn.run(app, host="127.0.0.1", port=a.port)
