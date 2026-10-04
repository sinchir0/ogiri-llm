"""picks.jsonl の集計: どのモデル/方向性が選ばれやすいか。  python -m ogiri.stats [--picks data/picks_vlm.jsonl]
選択率 = 選ばれた回数 / 提示された回数 (ランダムなら 1/提示数)
"""
import argparse
from collections import Counter

from .common import DATA, read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default=str(DATA / "picks.jsonl"))
    picks = read_jsonl(ap.parse_args().picks)
    if not picks:
        print("no picks yet")
        return
    K = sum(len(p["shown"]) for p in picks) / len(picks)  # 平均提示数
    none = sum(p["choice"] < 0 for p in picks)
    print(f"提示 {len(picks)} 件 / 全部微妙 {none} ({none / len(picks):.0%})")
    for key in ("model", "style"):
        shown, won = Counter(), Counter()
        for p in picks:
            for i, s in enumerate(p["shown"]):
                shown[s[key]] += 1
                won[s[key]] += i == p["choice"]
        print(f"\n[{key}] 選択率 (期待値 {1 / K:.1%})")
        for k in sorted(shown, key=lambda k: -won[k] / shown[k]):
            print(f"  {k:8s} {won[k] / shown[k]:6.1%}  ({won[k]}/{shown[k]})")
    shown, won = Counter(), Counter()
    for p in picks:
        for i, s in enumerate(p["shown"]):
            k = (s["model"], s["style"])
            shown[k] += 1
            won[k] += i == p["choice"]
    print("\n[model x style]")
    for k in sorted(shown, key=lambda k: -won[k] / shown[k]):
        print(f"  {k[0]:5s} {k[1]:8s} {won[k] / shown[k]:6.1%}  ({won[k]}/{shown[k]})")


if __name__ == "__main__":
    main()
