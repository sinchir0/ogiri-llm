"""大喜利Arena (pick-best 方式): 1お題につき6案(モデル/方向性は匿名)を見せ、一番面白いものを選ぶ。
「全部微妙」も記録する。選ばれた案 vs 残りからの選好ペアが prefs.jsonl に流れる。

事前に data/arena_pool.jsonl ({"topic","model","style","answer"}) を用意する(generate_styled.py)。
起動: python -m ogiri.arena --port 17080
記録: data/picks.jsonl (全ての提示と選択), data/prefs.jsonl (人間ペア)
"""
import argparse
import random
import time
from collections import Counter, defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .common import DATA, read_jsonl, write_jsonl

K = 6
app = FastAPI()
pool = defaultdict(list)  # topic -> [{model,style,answer}]
votes = Counter()  # topic -> 回答済み数(少ないお題を優先)
tickets = {}  # id -> (topic, shown)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI大喜利Arena</title><style>body{font:16px system-ui;max-width:680px;margin:1.5rem auto;padding:0 1rem}
button{width:100%;padding:.9rem;margin:.3rem 0;font-size:1.02rem;cursor:pointer;text-align:left}
.sub button{width:auto;display:inline-block;text-align:center;padding:.6rem 1rem}h2{margin:1rem 0}#c{color:#888;font-size:.9rem}</style>
<div id=c></div><h2 id=t></h2><div id=b></div>
<div class=sub><button id=n>全部微妙</button> <button id=s>スキップ</button></div>
<script>
let id,cnt=0;
async function load(){const r=await (await fetch('/api/set')).json();id=r.id;t.textContent='お題: '+r.topic;
b.innerHTML='';r.answers.forEach((a,i)=>{const e=document.createElement('button');e.textContent=a;e.onclick=()=>vote(i);b.appendChild(e)});
c.textContent='回答済み '+cnt+' 件'}
async function vote(i){await fetch('/api/vote',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id,choice:i})});cnt++;load()}
n.onclick=()=>vote(-1);s.onclick=load;load()
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/set")
def get_set():
    least = min(votes[t] for t in pool)
    topic = random.choice([t for t in pool if votes[t] == least])
    items = pool[topic][:]
    random.shuffle(items)
    shown, seen = [], set()
    for it in items:  # 同一回答は避ける
        if it["answer"] not in seen:
            seen.add(it["answer"])
            shown.append(it)
        if len(shown) == K:
            break
    tid = str(time.time_ns())
    tickets[tid] = (topic, shown)
    return {"id": tid, "topic": topic, "answers": [s["answer"] for s in shown]}  # モデル名/方向性は返さない


class Vote(BaseModel):
    id: str
    choice: int  # 0..K-1 / -1 = 全部微妙


@app.post("/api/vote")
def vote(v: Vote):
    t = tickets.pop(v.id, None)
    if not t or not -1 <= v.choice < K:
        return {"ok": False}
    topic, shown = t
    votes[topic] += 1
    write_jsonl(DATA / "picks.jsonl", [{"topic": topic, "shown": shown, "choice": v.choice}], "a")
    if v.choice >= 0:
        best = shown[v.choice]
        others = [s for i, s in enumerate(shown) if i != v.choice]
        pairs = [{"topic": topic, "chosen": best["answer"], "rejected": o["answer"], "source": "human"}
                 for o in random.sample(others, min(2, len(others)))]
        write_jsonl(DATA / "prefs.jsonl", pairs, "a")
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "arena_pool.jsonl"))
    ap.add_argument("--port", type=int, default=17080)
    a = ap.parse_args()
    for r in read_jsonl(a.pool):
        pool[r["topic"]].append(r)
    for r in read_jsonl(DATA / "picks.jsonl"):  # 再起動しても偏らないよう既存の回答数を引き継ぐ
        votes[r["topic"]] += 1
    uvicorn.run(app, host="127.0.0.1", port=a.port)
