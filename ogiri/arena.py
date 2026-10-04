"""大喜利Arena (pick-best 方式): 1お題につき複数案(モデル/方向性は匿名)を見せ、一番面白いものを選ぶ。
「全部微妙」も記録する。選ばれた案 vs 残りからの選好ペアが prefs.jsonl に流れる。

お題は 3 種類に対応する:
  テキストのみ   {"topic": "...",  "image": null}
  画像のみ       {"topic": null,   "image": "images/clot/xxx.jpg"}
  画像+テキスト  {"topic": "...",  "image": "images/clot/xxx.jpg"}
image は DATA 相対 (または絶対) パス。

事前に data/arena_pool.jsonl ({"topic","image","model","style","answer"}) を用意する(generate_styled.py 等)。
起動: python -m ogiri.arena --port 17080
記録: data/picks.jsonl (全ての提示と選択), data/prefs.jsonl (人間ペア; 画像お題は image 付き)
"""
import argparse
import random
import time
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .common import DATA, read_jsonl, topic_hash, write_jsonl

K = 6  # 提示数(--k で変更)
PICKS = DATA / "picks.jsonl"
PREFS = DATA / "prefs.jsonl"
PAIRS = 2  # 1回の選択から作る選好ペア数(--pairs)
PER_MODEL = False  # True: お題ごとに model(条件)別に1案ずつ提示する
KINDS = {"text", "image", "both"}  # 出題する種類(--kinds で絞る)
app = FastAPI()
pool = {}  # key(topic_hash) -> {"topic","image","items":[{model,style,answer,...}]}
votes = Counter()  # key -> 回答済み数(少ないお題を優先)
tickets = {}  # id -> (key, shown)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI大喜利Arena</title><style>body{font:16px system-ui;max-width:680px;margin:1.5rem auto;padding:0 1rem}
button{width:100%;padding:.9rem;margin:.3rem 0;font-size:1.02rem;cursor:pointer;text-align:left}
.sub button{width:auto;display:inline-block;text-align:center;padding:.6rem 1rem}h2{margin:1rem 0}#c{color:#888;font-size:.9rem}
#img[hidden]{display:none}#img{display:block;max-width:100%;max-height:60vh;margin:1rem auto;border-radius:6px}</style>
<div id=c></div><img id=img hidden alt="お題の画像(読み込み失敗)" onerror="t.textContent+=' [画像を読み込めません: ページを再読み込みしてください]'"><h2 id=t></h2><div id=b></div>
<div class=sub><button id=n>全部微妙</button> <button id=s>スキップ</button></div>
<script>
let id,cnt=0;
async function load(){const r=await (await fetch('/api/set')).json();id=r.id;
t.textContent=r.topic?'お題: '+r.topic:(r.image_url?'この画像で一言':'');
if(r.image_url){img.src=r.image_url;img.hidden=false}else{img.hidden=true;img.removeAttribute('src')}
b.innerHTML='';r.answers.forEach((a,i)=>{const e=document.createElement('button');e.textContent=a;e.onclick=()=>vote(i);b.appendChild(e)});
c.textContent='回答済み '+cnt+' 件'}
async function vote(i){await fetch('/api/vote',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id,choice:i})});cnt++;load()}
n.onclick=()=>vote(-1);s.onclick=load;load()
</script>"""


def kind_of(topic, image):
    return "both" if topic and image else "image" if image else "text"


def resolve_image(path):
    p = Path(path)
    return p if p.is_absolute() else DATA / p


@app.get("/")
def index():
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})  # 古いJSのまま動き続けるのを防ぐ


@app.get("/api/img/{key}")
def get_img(key: str):
    """プールに登録された画像だけを返す(任意パスの読み出しを防ぐため key で引く)。"""
    e = pool.get(key)
    if not e or not e["image"]:
        raise HTTPException(404)
    return FileResponse(resolve_image(e["image"]), headers={"Cache-Control": "max-age=3600"})


@app.get("/api/set")
def get_set():
    least = min(votes[k] for k in pool)
    key = random.choice([k for k in pool if votes[k] == least])
    e = pool[key]
    items = e["items"][:]
    random.shuffle(items)
    shown, seen = [], set()
    for it in items:  # 同一回答は避ける。PER_MODEL なら同一model(条件)は1案まで
        k = it["model"] if PER_MODEL else it["answer"]
        if k not in seen and it["answer"] not in {s["answer"] for s in shown}:
            seen.add(k)
            shown.append(it)
        if len(shown) == K:
            break
    random.shuffle(shown)  # 提示順の偏りを避ける
    tid = str(time.time_ns())
    tickets[tid] = (key, shown)
    return {"id": tid, "topic": e["topic"], "image_url": f"/api/img/{key}" if e["image"] else None,
            "answers": [s["answer"] for s in shown]}  # モデル名/方向性は返さない


class Vote(BaseModel):
    id: str
    choice: int  # 0..提示数-1 / -1 = 全部微妙


@app.post("/api/vote")
def vote(v: Vote):
    t = tickets.pop(v.id, None)
    if not t or not -1 <= v.choice < len(t[1]):
        return {"ok": False}
    key, shown = t
    e = pool[key]
    votes[key] += 1
    write_jsonl(PICKS, [{"topic": e["topic"], "image": e["image"], "shown": shown, "choice": v.choice}], "a")
    if v.choice >= 0:
        best = shown[v.choice]
        others = [s for i, s in enumerate(shown) if i != v.choice]
        pairs = []
        for o in random.sample(others, min(PAIRS, len(others))):
            p = {"topic": e["topic"], "chosen": best["answer"], "rejected": o["answer"], "source": "human"}
            if e["image"]:
                p["image"] = e["image"]  # 画像お題はDPOで画像も要るので残す
            if "think" in best and "think" in o:  # 思考つきモデル: DPOで 思考+回答 全体を比較するため残す
                p["chosen_think"], p["rejected_think"] = best["think"], o["think"]
            pairs.append(p)
        write_jsonl(PREFS, pairs, "a")
    return {"ok": True}


def load_pool(path):
    skipped = Counter()
    for r in read_jsonl(path):
        topic, image = (r.get("topic") or None), (r.get("image") or None)
        if not (topic or image):
            skipped["お題なし"] += 1
            continue
        if kind_of(topic, image) not in KINDS:
            skipped["--kinds 対象外"] += 1
            continue
        if image and not resolve_image(image).is_file():
            skipped["画像ファイルなし"] += 1
            continue
        e = pool.setdefault(topic_hash(topic, image), {"topic": topic, "image": image, "items": []})
        e["items"].append({**r, "topic": topic, "image": image})
    kinds = Counter(kind_of(e["topic"], e["image"]) for e in pool.values())
    print(f"pool: {len(pool)} お題 (テキスト{kinds['text']} / 画像{kinds['image']} / 画像+テキスト{kinds['both']})"
          + (f"  skipped: {dict(skipped)}" if skipped else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "arena_pool.jsonl"))
    ap.add_argument("--port", type=int, default=17080)
    ap.add_argument("--k", type=int, default=6, help="1お題で提示する案数")
    ap.add_argument("--picks", default=str(DATA / "picks.jsonl"), help="投票ログの出力先")
    ap.add_argument("--per-model", action="store_true", help="お題ごとに model(条件)別に1案ずつ提示(条件間比較用)")
    ap.add_argument("--kinds", nargs="+", default=sorted(KINDS), choices=sorted(KINDS),
                    help="出題するお題の種類: text(テキストのみ) image(画像のみ) both(画像+テキスト)")
    ap.add_argument("--prefs", default=str(DATA / "prefs.jsonl"), help="選好ペアの出力先")
    ap.add_argument("--pairs", type=int, default=2, help="1回の選択から作る選好ペア数(選ばれた案 vs 他の案)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    K, PICKS, PER_MODEL, KINDS = a.k, Path(a.picks), a.per_model, set(a.kinds)
    PREFS, PAIRS = Path(a.prefs), a.pairs
    load_pool(a.pool)
    if not pool:
        raise SystemExit("プールが空です(お題なし/画像なし/--kinds で全て除外)")
    for r in read_jsonl(PICKS):  # 再起動しても偏らないよう既存の回答数を引き継ぐ
        votes[topic_hash(r.get("topic"), r.get("image"))] += 1
    uvicorn.run(app, host=a.host, port=a.port)
