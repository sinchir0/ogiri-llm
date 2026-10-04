import hashlib
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("OGIRI_DATA", Path(__file__).resolve().parent.parent / "data"))
BASE_MODEL = os.environ.get("OGIRI_BASE", "Qwen/Qwen3.5-9B")
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


def topic_hash(topic: str | None, image: str | None = None) -> str:
    key = f"img:{image}" if image else (topic or "").strip()
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def chat(topic: str | None = None, answer: str | None = None, image: str | None = None, system: str = SYSTEM):
    """contentは全role常にblock形式(VLM用)。topicがNoneなら画像のみのお題として扱う。
    (Datasetに混在させる際、role間/行間でcontentの型(str/list)が割れるとpyarrowが落ちるため常にlistにする)"""
    text = f"お題: {topic}" if topic else "この画像で大喜利に回答してください。"
    content = ([{"type": "image", "image": image}] if image else []) + [{"type": "text", "text": text}]
    m = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m
