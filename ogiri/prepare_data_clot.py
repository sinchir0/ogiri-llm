"""公開データ zhongshsh/CLoT-Oogiri-GO (日本語, jp.jsonl) から画像入力の SFT/選好データを作る。
python -m ogiri.prepare_data_clot

- jp.jsonl は T2T/I2T/IT2T の全件が画像付き(T2TもIT2Tもお題文/お題画像が画像内に焼き込まれている形式)。
  テキストのみのお題は含まれないため、全件を「画像お題」として扱う(topic=None, image=パス)。
  埋め込み文字の読み取りはVLM自身のOCR能力に委ねる(別途OCR抽出はしない)。
- star(いいね数)はお題画像の人気度に依存し絶対値では比較できないため、
  同一画像(=同一お題)グループ内でのパーセンタイルで win/ok/bad を決める。
- 画像は data/images/clot/<id>.jpg に展開する。
※ CC BY 4.0。Bokete由来の画像は著作権懸念のあるものが混在し得るため研究目的に限ること。
"""
import argparse
import collections
import json
import random
import zipfile

from huggingface_hub import hf_hub_download

from .benchmark import filter_train
from .common import DATA, write_jsonl

REPO = "zhongshsh/CLoT-Oogiri-GO"
IMG_DIR = DATA / "images" / "clot"


def _star(r):
    return int(str(r["star"]).replace(",", ""))


def load_rows():
    path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename="jp.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r.get("image") and (r.get("text") or "").strip()]


def extract_images(ids):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename="images.zip")
    with zipfile.ZipFile(zip_path) as z:
        for i in ids:
            dst = IMG_DIR / f"{i}.jpg"
            if not dst.exists():
                with z.open(f"images/{i}.jpg") as src, open(dst, "wb") as out:
                    out.write(src.read())


def build(rows, win_frac=0.3, bad_frac=0.3, min_star_for_win=2, seed=0):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["image"]].append(r)
    rng = random.Random(seed)
    sft, prefs = [], []
    for img, rs in by.items():
        rs = sorted(rs, key=lambda r: -_star(r))
        n = len(rs)
        win_n = max(1, round(n * win_frac))
        bad_n = max(1, round(n * bad_frac))
        rel_path = f"images/clot/{img}.jpg"
        for idx, r in enumerate(rs):
            star = _star(r)
            if idx < win_n and star >= min_star_for_win:
                tier = "win"
            elif idx >= n - bad_n:
                tier = "bad"
            else:
                tier = "ok"
            sft.append({"topic": None, "image": rel_path, "answer": r["text"].strip(),
                        "tier": tier, "star": star, "clot_type": r["type"]})
        top = [r for r in rs[:win_n] if _star(r) >= min_star_for_win]
        bot = rs[-bad_n:]
        for h in top:
            cands = [l for l in bot if l is not h]
            for l in rng.sample(cands, min(2, len(cands))):
                prefs.append({"topic": None, "image": rel_path, "chosen": h["text"].strip(),
                              "rejected": l["text"].strip(), "source": "clot"})
    return sft, prefs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--win_frac", type=float, default=0.3)
    p.add_argument("--bad_frac", type=float, default=0.3)
    p.add_argument("--min_star_for_win", type=int, default=2)
    p.add_argument("--sft_out", default=str(DATA / "sft_clot.jsonl"))
    p.add_argument("--prefs_out", default=str(DATA / "prefs_clot.jsonl"))
    a = p.parse_args()

    rows = load_rows()
    extract_images(sorted({r["image"] for r in rows}))
    sft, prefs = build(rows, a.win_frac, a.bad_frac, a.min_star_for_win)
    sft, prefs = filter_train(sft), filter_train(prefs)  # ベンチお題とのリーク防止(将来の画像ベンチ用)
    write_jsonl(a.sft_out, sft)
    write_jsonl(a.prefs_out, prefs)
    tiers = collections.Counter(r["tier"] for r in sft)
    print(f"rows={len(rows)} images={len({r['image'] for r in rows})} "
          f"sft={len(sft)} ({dict(tiers)}) prefs={len(prefs)}")
