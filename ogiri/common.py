import hashlib
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("OGIRI_DATA", Path(__file__).resolve().parent.parent / "data"))
BASE_MODEL = os.environ.get("OGIRI_BASE", "Qwen/Qwen3-8B")
JUDGE_MODEL = os.environ.get("OGIRI_JUDGE", BASE_MODEL)

SYSTEM = (
    "あなたは大喜利の達人です。お題に対し、意外性があり、短く、お題に的確に絡み、"
    "既視感のない回答を1つだけ出力してください。説明や前置きは不要です。"
)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def topic_hash(topic: str) -> str:
    return hashlib.sha1(topic.strip().encode()).hexdigest()[:12]


def chat(topic: str, answer: str | None = None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"お題: {topic}"}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m
