"""大喜利デモUI: テキストのみ / 画像のみ / 画像+テキスト を入力して、候補(思考つき)を生成する。
vLLM(複数LoRA切替)で推論する。1つのモデルが3種類の入力すべてを処理する。

起動例:
  python -m ogiri.demo --lora sft=ckpt/sft_proc --lora dpo=ckpt/dpo_proc --port 17090
最初に指定した --lora が既定。--lora を省略するとベースモデルのみ。
"""
import argparse
import io
import threading

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from .common import BASE_MODEL
from .prompts import PROC_SYSTEM

app = FastAPI()
STATE = {"loras": {}}
LOCK = threading.Lock()  # vLLMエンジンは1リクエストずつ処理する
MAX_SIDE = 1280  # 大きすぎる画像は縮小する(トークン数の抑制)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>大喜利 生成デモ</title><style>
body{font:16px system-ui;max-width:760px;margin:1.5rem auto;padding:0 1rem}
section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
h2{margin:.2rem 0 1rem;font-size:1.1rem}
textarea,input[type=text],select{width:100%;box-sizing:border-box;font-size:1rem;padding:.5rem}
button{padding:.6rem 1.2rem;margin-top:.5rem;cursor:pointer}
.bar{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;background:#f6f6f6;padding:.6rem;border-radius:8px}
.bar label{display:flex;gap:.4rem;align-items:center}
.bar input[type=number]{width:4.5rem}
.bar select{width:auto}
.cand{margin-top:.6rem;padding:.6rem;background:#f6f6f6;border-radius:6px}
.cand b{font-size:1.05rem}
details{margin-top:.3rem;color:#555;font-size:.9rem;white-space:pre-wrap}
.msg{color:#888;margin-top:.8rem}
img.preview{max-width:100%;max-height:240px;display:block;margin-top:.5rem}
</style>
<h1>大喜利 生成デモ</h1>
<div class=bar>
<label>モデル <select id=model></select></label>
<label>候補数 <input type=number id=n value=4 min=1 max=8></label>
<label>温度 <input type=number id=temp value=0.7 min=0 max=1.5 step=0.1></label>
<label><input type=checkbox id=showthink> 思考を表示</label>
</div>

<section>
<h2>① テキストのみ</h2>
<input type=text id=t1 placeholder="お題を入力" value="こんなコンビニは嫌だ。どんなコンビニ？">
<button onclick="run(1)">生成</button>
<div id=o1></div>
</section>

<section>
<h2>② 画像のみ</h2>
<input type=file id=f2 accept="image/*" onchange="preview(2)"><img class=preview id=p2 hidden>
<button onclick="run(2)">生成</button>
<div id=o2></div>
</section>

<section>
<h2>③ 画像 + テキスト</h2>
<input type=file id=f3 accept="image/*" onchange="preview(3)"><img class=preview id=p3 hidden>
<input type=text id=t3 placeholder="お題を入力" style="margin-top:.5rem">
<button onclick="run(3)">生成</button>
<div id=o3></div>
</section>

<script>
const $=id=>document.getElementById(id);
fetch('/api/models').then(r=>r.json()).then(j=>{j.models.forEach(m=>{const o=document.createElement('option');o.value=o.textContent=m;$('model').appendChild(o)})});
function preview(n){const f=$('f'+n).files[0];if(!f)return;const p=$('p'+n);p.src=URL.createObjectURL(f);p.hidden=false}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function run(k){
  const out=$('o'+k), t=$('t'+k), f=$('f'+k);
  const hasT=!!(t&&t.value.trim()), hasI=!!(f&&f.files[0]);
  if(!hasT&&!hasI){out.innerHTML='<div class=msg>お題か画像を入力してください</div>';return}
  if(k==3&&!(hasT&&hasI)){out.innerHTML='<div class=msg>画像とテキストの両方を入力してください</div>';return}
  out.innerHTML='<div class=msg>生成中...</div>';
  const fd=new FormData();
  if(hasT)fd.append('topic',t.value.trim());
  if(hasI)fd.append('image',f.files[0]);
  fd.append('model',$('model').value);fd.append('n',$('n').value);fd.append('temperature',$('temp').value);
  try{
    const j=await (await fetch('/api/generate',{method:'POST',body:fd})).json();
    if(j.error){out.innerHTML='<div class=msg>エラー: '+esc(j.error)+'</div>';return}
    out.innerHTML=j.candidates.map(c=>'<div class=cand><b>'+esc(c.answer)+'</b>'+
      ($('showthink').checked&&c.think?'<details open><summary>思考</summary>'+esc(c.think)+'</details>':'')+'</div>').join('')
      ||'<div class=msg>有効な候補が得られませんでした(もう一度試してください)</div>';
  }catch(e){out.innerHTML='<div class=msg>エラー: '+esc(String(e))+'</div>'}
}
</script>"""


def parse_output(text):
    """'<think>\\n' の直後から始まる出力を (思考, 回答) に分ける。思考が閉じていなければ None。"""
    if "</think>" not in text:
        return None
    think, answer = text.split("</think>", 1)
    answer = answer.strip().split("\n")[0].strip()
    return (think.strip(), answer) if answer else None


def generate(topic, image, model_name, n, temperature):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok, llm = STATE["tok"], STATE["llm"]
    text = f"お題: {topic}" if topic else "この画像で大喜利に回答してください。"
    content = [{"type": "image"}, {"type": "text", "text": text}] if image is not None else text
    prompt = tok.apply_chat_template([{"role": "system", "content": PROC_SYSTEM}, {"role": "user", "content": content}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
    req = {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
    lora = STATE["loras"].get(model_name)
    sp = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=500)
    with LOCK:
        out = llm.generate([req], sp, lora_request=LoRARequest(model_name, lora[0], lora[1]) if lora else None,
                           use_tqdm=False)[0]
    cands, seen = [], set()
    for c in out.outputs:
        r = parse_output(c.text)
        if r and r[1] not in seen:
            seen.add(r[1])
            cands.append({"think": r[0], "answer": r[1]})
    return cands


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


@app.get("/api/models")
def models():
    return {"models": list(STATE["loras"]) or ["base"]}


@app.post("/api/generate")
async def api_generate(topic: str | None = Form(None), image: UploadFile | None = File(None),
                       model: str = Form(""), n: int = Form(4), temperature: float = Form(0.7)):
    try:
        topic = (topic or "").strip() or None
        img = None
        if image is not None and image.filename:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(await image.read()))).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE))
            w, h = img.size
            if max(w, h) / max(1, min(w, h)) > 150:  # Qwen系processorのアスペクト比上限(200)に対する安全マージン
                return JSONResponse({"error": "画像の縦横比が極端です"}, status_code=400)
        if topic is None and img is None:
            return JSONResponse({"error": "お題か画像が必要です"}, status_code=400)
        n = max(1, min(int(n), 8))
        cands = generate(topic, img, model, n, max(0.0, min(float(temperature), 1.5)))
        return {"candidates": cands}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--lora", action="append", default=[], help="名前=パス (複数可)。例 dpo=ckpt/dpo_proc")
    ap.add_argument("--port", type=int, default=17090)
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    a = ap.parse_args()

    from vllm import LLM

    for i, spec in enumerate(a.lora, 1):
        name, path = spec.split("=", 1)
        STATE["loras"][name] = (i, path)
    STATE["llm"] = LLM(model=a.base, enable_lora=bool(a.lora), max_lora_rank=64, max_loras=max(1, len(a.lora)),
                       max_model_len=4096, gpu_memory_utilization=a.gpu_mem, limit_mm_per_prompt={"image": 1})
    STATE["tok"] = STATE["llm"].get_tokenizer()
    uvicorn.run(app, host="127.0.0.1", port=a.port)
