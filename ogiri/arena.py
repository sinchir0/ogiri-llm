"""AI大喜利Arena: 匿名A/Bを見せて「どっちが面白い？」を集める。投票は battles.jsonl と prefs.jsonl に流れる。

事前に data/arena_pool.jsonl に {"topic","model","answer"} を溜めておく(bench お題 × 各モデル)。
起動: python -m ogiri.arena --port 17080
"""
import argparse
import random
import time
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .common import DATA, read_jsonl, write_jsonl

app = FastAPI()
pool = defaultdict(list)  # topic -> [(model, answer)]
tickets = {}  # id -> (topic, a_model, a, b_model, b)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI大喜利Arena</title><style>body{font:16px system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}
button{width:100%;padding:1rem;margin:.4rem 0;font-size:1.05rem;cursor:pointer}h2{margin:1.5rem 0}</style>
<h2 id=t></h2><button id=a></button><button id=b></button><button id=n>どちらも微妙</button><button id=s>スキップ</button>
<script>
let id;async function load(){const r=await (await fetch('/api/pair')).json();id=r.id;
t.textContent='お題: '+r.topic;a.textContent=r.a;b.textContent=r.b}
async function vote(w){await fetch('/api/vote',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id,winner:w})});load()}
a.onclick=()=>vote('a');b.onclick=()=>vote('b');n.onclick=()=>vote('tie');s.onclick=load;load()
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/pair")
def pair():
    topic = random.choice([t for t, v in pool.items() if len({m for m, _ in v}) >= 2])
    (am, a), (bm, b) = _two_models(topic)
    tid = f"{time.time_ns()}"
    tickets[tid] = (topic, am, a, bm, b)
    return {"id": tid, "topic": topic, "a": a, "b": b}  # モデル名は返さない(匿名)


def _two_models(topic):
    by = defaultdict(list)
    for m, ans in pool[topic]:
        by[m].append(ans)
    ms = random.sample(list(by), 2)
    return [(m, random.choice(by[m])) for m in ms]


class Vote(BaseModel):
    id: str
    winner: str


@app.post("/api/vote")
def vote(v: Vote):
    t = tickets.pop(v.id, None)
    if not t or v.winner not in ("a", "b", "tie"):
        return {"ok": False}
    topic, am, a, bm, b = t
    write_jsonl(DATA / "battles.jsonl", [{"topic": topic, "a_model": am, "b_model": bm, "a": a, "b": b, "winner": v.winner}], "a")
    if v.winner != "tie":
        c, r = (a, b) if v.winner == "a" else (b, a)
        write_jsonl(DATA / "prefs.jsonl", [{"topic": topic, "chosen": c, "rejected": r, "source": "human"}], "a")
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "arena_pool.jsonl"))
    ap.add_argument("--port", type=int, default=17080)
    a = ap.parse_args()
    for r in read_jsonl(a.pool):
        pool[r["topic"]].append((r["model"], r["answer"]))
    uvicorn.run(app, host="127.0.0.1", port=a.port)
