# 最終的な学習手法 (手順SFT → 人間選好DPO)

**1つのモデル (Qwen3.5-9B + LoRA) で、テキストのみ / 画像のみ / 画像+テキスト の3種類のお題すべてに回答する。**
回答の前に、大喜利の考え方の手順を日本語5行で `<think>` に書かせ、そのあとで回答を1つ出す。

```
お題 (テキスト / 画像 / 画像+テキスト)
  └─ <think> 1.ありがち 2.ヒンジ 3.遠い世界 4.衝突と理屈 5.具体化 </think>
       └─ 回答 (1つ、短く)
```

学習は2段階:

1. **手順SFT** — 人間に高評価された回答に、「その回答に至る手順(思考)」を付けて学習 (`ogiri/make_rationale.py` → `ogiri/sft.py --proc`)
2. **人間選好DPO** — SFT済みモデルの4案から人間が選んだ「面白い案」を chosen、他を rejected として DPO (`ogiri/arena.py` → `ogiri/dpo_proc.py`)

## 1. 背景: なぜこの形にしたか

| 段階 | 結果 |
|---|---|
| Qwen3-8B (テキストのみ) + SFT | 人間の A/B 投票で base と SFT に有意差なし。71%が引き分け、6案から選ぶ形式でも約7割が「全部微妙」 |
| Qwen3.5-9B (VLM) + SFT (回答のみ, 27,267件) | 日本語の短い回答は出るが、面白さは頭打ち |
| 手順プロンプトの比較 (下表) | 画像では手順プロンプトが有効、テキストでは差が小さい |

手順プロンプトの条件比較 (人間が4案から1つ選ぶ。テキスト100問 + 画像41件のプールから105提示、選ばれたのは50件):

| 条件 | 内容 | テキストで選ばれた数 | 画像系で選ばれた数 |
|---|---|---|---|
| A | SFT + 従来の短い SYSTEM (思考なし) | 11 | 0 |
| B | ベース + 手順プロンプト全文 (思考あり) | 6 | **11** |
| C | SFT + 圧縮した手順 (思考なし) | 5 | 5 |
| D | ベース + 短い CoT (思考あり) | **12** | 0 |

- 「全部微妙」は 52%。サンプルが少なく、テキストの差は偶然の範囲。画像系の B の強さ (16件中11件) は偶然とは考えにくい。
- ただし B/D は **ベースモデル** (LoRA なし) の思考モード。思考が英語で延々と案を列挙し、予算 (2048トークン) を全件で超過 → `</think>` を強制挿入して回答させていた。
- 「1モデルで全入力を処理する」要件のため、手順を **SFT で内在化**し、短い日本語の思考を自前で書けるモデルにする方針に切り替えた。

## 2. 手順SFT

### 2.1 プロンプト (学習・推論で共通)
`ogiri/prompts.py` の `PROC_SYSTEM`:

```
あなたは大喜利の達人です。回答の前に、<think> の中で、日本語で次の手順を各1行、合計5行以内で考えてください。
1. ありがち: 観客が予想する答えを挙げる(回答には使わない)
2. ヒンジ: 意味をずらせる語・物・状況
3. 遠い世界: 意外な別領域(会社・役所・ゲーム・学校など)
4. 衝突と理屈: 二つの世界を、文字通り解釈・役割逆転・誇張・偽の因果などで一本の理屈でつなぐ
5. 具体化: 映像が浮かぶ細部にして、不要な語を削る
考え終えたら、最も面白い回答を1つだけ、短く出力してください。説明や前置きは不要です。
```

ユーザー側: テキスト=`お題: {お題}` / 画像のみ=`この画像で大喜利に回答してください。`(画像つき) / 画像+テキスト=画像 + `お題: {お題}`。

### 2.2 教師データ (思考の作り方)
公開データには回答はあるが思考がない。**人間に高評価された回答から逆算して**、ベースモデル (Qwen3.5-9B, LoRAなし, 思考オフ) に5行の思考を書かせた (`ogiri/make_rationale.py`)。

- 元データ:
  - テキスト: `iammytoo/japanese-humor-evaluation-v2` の score ≥ 3.5 (4,407件)
  - 画像: `zhongshsh/CLoT-Oogiri-GO` 日本語の win tier (同一画像内の star パーセンタイル上位) のうち、回答60字以下の3,600件をランダム抽出
- 分析者プロンプト + 5行形式 + 例1つ。出力の書き出しを `1. ありがち:` に固定して形式を安定させる。温度0.5。
- 検証して不合格を除外: 5行ちょうど / 各行が `N. ラベル:` で始まる / 70字以内 / 他言語文字なし / 8字以上の回答の丸写しなし。
- 結果: **7,251件が有効** (テキスト4,081 + 画像3,170、有効率 90.6%)。

> **限界**: 思考は答えが先にあって後から書いた説明 (rationalization) で、本物の発想過程ではない。形式の検証しかしておらず、中身の正しさ (特に画像の読み取り) は保証していない。

### 2.3 学習設定 (`ogiri/sft.py --proc`)
| 項目 | 値 |
|---|---|
| ベース | Qwen3.5-9B (bf16。QLoRAは公式非推奨のため不使用) |
| LoRA | r=32, α=64, dropout 0.05, 対象 q/k/v/o_proj, gate/up/down_proj (DeltaNet固有層・vision towerは対象外) |
| 最適化 | lr 1e-4, cosine, warmup 3%, 1 epoch |
| バッチ | 8 × grad_accum 4 = 32、227ステップ |
| 損失 | プロンプト部 (`<think>\n` まで) を `-100` でマスクし、**思考 + 回答**にのみ損失 |
| 高速化 | `flash-linear-attention` (DeltaNetのTritonカーネル)。勾配チェックポイントは必須 (OFFだとバッチ8でもOOM) |
| 結果 | train_loss 1.07、約37分 (RTX PRO 5000 Blackwell 48GB) |

学習の系列 (assistant 部): `<think>\n{5行}\n</think>\n\n{回答}<|im_end|>`

## 3. 人間選好の収集

- SFT済みモデルで 141 お題 (テキスト100 / 画像のみ31 / 画像+テキスト10) × 4案を生成 (温度0.7, top_p 0.95, 思考も保存)。
- 画像は CC0 の `ThePioneer/japanese-photos` から、ボケどころのある写真を選んだ (`data/image_topics.jsonl`)。学習済み画像との重複・リークを避けるため CLoT の画像は使わない。
- アリーナ (`ogiri/arena.py`) で、お題ごとに4案を匿名・ランダム順で提示し、人間が1つ選ぶ (「全部微妙」あり)。
- 結果: 105提示、**全部微妙 25%** (条件比較のときは 52%。ただし条件・提示形式が違うため厳密な比較ではない)。選ばれた位置はほぼ均等 (0:16 / 1:21 / 2:18 / 3:24)、chosen/rejected の回答長に差なし (11.4 / 11.8 字)。
- 選ばれた案 vs 他の全案 (最大3組) で選好ペアを作成: **230組・79お題**。ペアには**思考も保存**する (`chosen_think` / `rejected_think`)。

## 4. DPO (`ogiri/dpo_proc.py`)

TRL の DPOTrainer ではなく、SFT のコレーター相当を使った自作ループ (VLM・思考形式への対応のため)。

- 方策 = base + 手順SFT の LoRA をそのまま学習可能にして更新。参照モデルは別に持たず、**学習前に全ペアの参照 logp を1回だけ計算**して保持 (= 初期のSFTモデルが参照)。
- 比較するのは **思考 + 回答 の全体** の logp (画像ペアは画像つき)。
- 損失 = DPO(sigmoid) + α · NLL(chosen, トークン平均)。ペアが少ないため SFT 項で崩れを抑える (RPO風)。
- β=0.1, lr 2e-5, α=0.1, 3 epoch, 1ステップ=8ペア, 検証=お題単位で15%。
- 学習 197組 / 検証 33組 (11お題)。

| epoch | 学習 DPO loss / 正解率 | 検証 正解率 / margin |
|---|---|---|
| 0 (初期) | - | 0% / 0 (差なしが正常) |
| 1 | 0.524 / 81% | 67% / +0.157 |
| 2 | 0.133 / 100% | 61% / +0.228 |
| 3 | 0.051 / 100% | 61% / +0.288 |

> **解釈**: 学習データには完全に過学習 (正解率100%) している。検証は 61〜67% (33ペア) で、偶然 (50%) との差は有意とは言えない。**DPO が SFT より良いという証拠は、現時点ではない。** 確かめるには、新しいお題で SFT 版と DPO 版の出力を並べた追加の人間評価が必要。ペア数を増やすか、epoch を1〜2に抑える/ β を上げる余地がある。

## 4.5 システムプロンプトの設計と参考にした研究
システムプロンプトの手順 (ありがち → ヒンジ → 遠い世界 → 衝突と理屈 → 具体化) は、ユーモアの理論と、ユーモア生成の研究の要点をまとめた調査メモをもとに、作者が設計した。
**各論文の手法を再現したものではない**。論文の内容は概要・要旨のレベルで確認しており、手順への対応づけは作者の解釈を含む。

| 手順 | 参考にした考え方 | 出典 |
|---|---|---|
| 1. ありがち | 笑いは、観客の予測の形成と、その裏切りから生じる (不調和の検出)。予測を先に明示して回答から除く | Suls (1972) |
| 3. 遠い世界 / 4. 衝突と理屈 | 2つの意味世界の対立 (Script Opposition) と、それをつなぐ理屈 (Logical Mechanism) | Attardo & Raskin (1991) |
| 4. 衝突と理屈 | 不調和を「そういうことか」と解消できること (不調和の解消) | Suls (1972) |
| 4. 衝突と理屈 | 「おかしい (violation)」が「無害 (benign)」と同時に成り立つこと | McGraw & Warren (2010) |
| 2. ヒンジ / 5. 具体化 | 出典に基づかない、作者の設計 (ずらせる語を探す / 映像が浮かぶ細部にして削る) | - |

学習方針については、次の研究も参考にした。

- **段階的な発想の手順を踏む**: SemEval-2026 Task 1 (MWAHAHA、人間の比較評価でジョーク生成を競う) の参加システム RAGthoven は、Planner → Best-of-N Writer → Reflector → Judge の多段パイプラインで、ユーモア理論 (Benign Violation Theory など) に基づく。本モデルはこれを1回の出力内の短い思考 (5行) に圧縮した形で、推論時の Best-of-N や Judge は含まない。
- **理論に基づくデータ設計 → SFT**: HumorGen は、心理学のユーモア理論に基づく複数のペルソナで合成データを作って 7B を SFT する。同論文は、DPO と O-GRPO が SFT を上回らなかったと報告している。本モデルでも、DPO が SFT を上回る証拠は得られていない (検証33ペアで有意差なし)。
- **大喜利データと LoT**: Oogiri-GO データセットと Leap-of-Thought (CLoT) は、本モデルの学習データ (画像お題) の出典。

### 参考文献
- Attardo, S., & Raskin, V. (1991). Script theory revis(it)ed: Joke similarity and joke representation model. *HUMOR: International Journal of Humor Research*, 4(3-4). (General Theory of Verbal Humor)
- McGraw, A. P., & Warren, C. (2010). [Benign violations: Making immoral behavior funny](https://leeds-faculty.colorado.edu/mcgrawp/pdf/mcgraw.warren.2010.pdf). *Psychological Science*, 21(8), 1141-1149.
- Suls, J. M. (1972). A two-stage model for the appreciation of jokes and cartoons: An information-processing analysis. In Goldstein, J. H., & McGhee, P. E. (Eds.), [*The Psychology of Humor*](https://shop.elsevier.com/books/the-psychology-of-humor/goldstein/9780122889509). Academic Press.
- Zhong, S., Huang, Z., Gao, S., Wen, W., Lin, L., Zitnik, M., & Zhou, P. (2024). [Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation](https://arxiv.org/abs/2312.02439). *CVPR 2024*. (CLoT / Oogiri-GO)
- Ajayi, E., & Mitra, P. (2026). [HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation](https://arxiv.org/abs/2604.09629). arXiv:2604.09629.
- [RAGthoven at SemEval-2026 Task 1: A Multi-Stage Pipeline Walks Into a Benchmark and Barely Clears the Bar](https://arxiv.org/pdf/2607.13189). arXiv:2607.13189.
- [SemEval 2026 Task 1: MWAHAHA — Models Write Automatic Humor And Humans Annotate](https://www.aclweb.org/portal/node/14303).

## 5. 既知の限界
- 評価者は1名。投票数は約100件で、統計的な結論は出せない。
- 思考は後付けの説明であり、思考の質が回答の質を押し上げているかは未検証 (思考なしSFTとのアブレーションなし)。
- 学習データの出典: iammytoo データは NHK 番組由来で権利が未確認、CLoT-Oogiri-GO の画像は Bokete 由来 (CC BY 4.0 だがサイト規約に従う)。**研究目的に限る。** そのため `data/sft_proc.jsonl` などの学習データはリポジトリに含めない。
- 生成の不適切な内容は除外していない (投票で選ばれない限り学習に入らないが、モデル自体は出し得る)。

## 6. 再現手順
```bash
pip install -r requirements.txt flash-linear-attention
python -m ogiri.prepare_data               # テキスト SFT データ
python -m ogiri.prepare_data_clot          # 画像 SFT データ (CLoT)
python -m ogiri.make_rationale --n_text 4407 --n_img 3600 --out data/sft_proc.jsonl   # 思考の逆算
python -m ogiri.sft --proc data/sft_proc.jsonl --epochs 1 --batch_size 8 --grad_accum 4 --save_steps 50 --out ckpt/sft_proc
python -m ogiri.generate_prompted --lora ckpt/sft_proc --items <お題jsonl> --conds H_proc --n 4 --out data/arena_pool_proc.jsonl
python -m ogiri.arena --pool data/arena_pool_proc.jsonl --picks data/picks_proc.jsonl --prefs data/prefs_proc.jsonl --k 4 --pairs 3
python -m ogiri.dpo_proc --init ckpt/sft_proc --prefs data/prefs_proc.jsonl --out ckpt/dpo_proc --epochs 3
python -m ogiri.demo --lora sft=ckpt/sft_proc --lora dpo=ckpt/dpo_proc     # デモUI
```
Blackwell (sm120) では `VLLM_USE_FLASHINFER_SAMPLER=0` が必要 (FlashInfer サンプラーが失敗する)。

## 7. デモUI (`ogiri/demo.py`)
テキストのみ / 画像のみ / 画像+テキスト を入力し、SFT・DPO を切り替えて、候補数・温度・思考の表示を指定できる。vLLM + 複数LoRA で、1回の生成は約3秒。

## 8. モデルの重み
Hugging Face Hub: [`sinchir0/ogiri-qwen3.5-9b-proc`](https://huggingface.co/sinchir0/ogiri-qwen3.5-9b-proc)
- ルート: DPO 後の LoRA アダプタ (`dpo_proc`)
- `sft_proc/`: 手順SFT 後の LoRA アダプタ
- ベースは `Qwen/Qwen3.5-9B`。推論時は上記の `PROC_SYSTEM` と `enable_thinking=True` を使い、`</think>` 以降を回答として取り出す。

## 9. 環境
RTX PRO 5000 Blackwell 48GB (sm120) / PyTorch 2.13 (cu130) / vLLM 0.29 / transformers 5.18 / TRL 1.14 (SFT) / PEFT 0.21 / flash-linear-attention 0.5.2。
