# ogiri-llm — 大喜利LLM開発基盤

データフライホイール型: `お題 → N案生成 → Judge/人間で選好 → SFT/DPO → 再生成 → …`

## パイプライン

| Step | コマンド | 内容 |
|---|---|---|
| 0 | `python -m ogiri.benchmark build` | 固定評価お題(500問)を切り出す。学習には絶対使わない |
| 1 | `python -m ogiri.sft` | 高品質 (お題,回答) JSONL で QLoRA SFT |
| 2 | `python -m ogiri.generate` | 各お題に N=64 案を vLLM で生成 |
| 3 | `python -m ogiri.judge` | 5軸(面白さ/意外性/適合/独創性/簡潔さ)で採点 → 選好ペア作成 |
| 4 | `python -m ogiri.dpo` | ペアで DPO (`--loss simpo` 等も可) |
| 5 | `python -m ogiri.bestofn` | 推論時 Best-of-N (Generator → Judge → 上位K) |
| 6 | `python -m ogiri.eval` | 対戦ログから Bradley–Terry / Elo 推定 |
| 7 | `python -m ogiri.arena` | 人間投票アリーナ(投票が次の選好データになる) |

## 最終的な学習手法 (手順SFT → 人間選好DPO)
テキスト / 画像 / 画像+テキスト を **1つのモデル** で扱い、回答の前に大喜利の手順 (日本語5行) を `<think>` に書かせる。
詳細・結果・限界は [docs/TRAINING.md](docs/TRAINING.md)。

| Step | コマンド | 内容 |
|---|---|---|
| a | `python -m ogiri.make_rationale` | 高評価回答から思考(5行)を逆算して教師データ作成 |
| b | `python -m ogiri.sft --proc data/sft_proc.jsonl` | 手順SFT (思考+回答に損失) |
| c | `python -m ogiri.generate_prompted --conds H_proc` | 4案生成 (思考つき) |
| d | `python -m ogiri.arena --k 4 --pairs 3 --prefs data/prefs_proc.jsonl` | 人間が選ぶ → 思考つき選好ペア |
| e | `python -m ogiri.dpo_proc --init ckpt/sft_proc` | 思考+回答 全体の DPO (画像対応) |
| f | `python -m ogiri.demo --lora sft=... --lora dpo=...` | デモUI (3種類の入力) |

## データ形式
- `data/sft.jsonl`: `{"topic": "...", "answer": "...", "tier": "win|ok|bad"}` (tier は任意。`win` のみ SFT に使用)
- `data/prefs.jsonl`: `{"topic","chosen","rejected","source":"judge|human"}`
- `data/battles.jsonl`: `{"topic","a_model","b_model","a","b","winner":"a|b|tie"}`

## 推奨順序 (元計画に対応)
1. ベンチ作成 → 2. Base(Qwen3 4B/8B Instruct)測定 → 3. SFT → 4. Best-of-32+Judge
→ 5. DPO/GRPO → 6. Arena で人間評価を継続収集。
Ablation: SFTのみ / SFT+DPO / SFT+RL / Best-of-N 単体 をすべて同一ベンチで battle。

## 注意
- Judge は LLM 自己採点でバイアスがある(長文・既視感を好む)。定期的に人間投票との相関を確認し、
  相関が低ければ人間ペアで Judge 自体を微調整する。
- 評価用お題は学習データと重複排除する(`benchmark.py` が hash で除外)。
- 著作権: 大喜利サイト等のスクレイピングは規約・権利を確認の上、自前収集/許諾データを使うこと。
