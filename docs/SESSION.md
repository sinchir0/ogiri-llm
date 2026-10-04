# セッションまとめ (2026-09-29)

## 目的
「世界一面白い大喜利LLM」を作る。方針は **データフライホイール**:
`お題 → N案生成 → 人間/Judgeが選好 → SFT/DPO → 再生成 → …`
単純なSFTより選好学習(DPO等)+ Best-of-N + 大喜利専用Judge を主軸にする。
評価は自動指標ではなく、匿名A/B対戦 → Bradley–Terry / Elo。

## 決定事項
- ベースモデル: Qwen3-8B (LoRA/QLoRA)
- お題: 私(Claude)が100問を作成 (`data/sample_topics.jsonl`)
- **出力の評価は人間(sinchir0)が行う**。Judge LLM の自己採点は補助扱い
- 分割: 学習90 / 評価10 (9:1)。評価お題は学習に使わない
- 生成数: 1問あたり3案
- リポジトリ: `sinchir0/ogiri-llm` (**private**)

## 実装済み (`ogiri/`)
benchmark(分割) / sft / generate(vLLM) / judge(5軸採点→選好ペア) / dpo /
bestofn / eval(Bradley–Terry) / arena(A/B投票UI、投票が prefs.jsonl に追記)

## 環境
- GPU: RTX PRO 5000 Blackwell 72GB (sm120)
- `.venv`: PyTorch 2.13 (cu130), vLLM 0.30, TRL 1.14, PEFT
- ディスク空きは約8GB。`/workspace` は永続ボリュームではない(recycle で消える)
- git identity: `sinchir0`、`gh` ログイン済み、`gh auth setup-git` 設定済み
- RTK (トークン節約プロキシ) 導入済み

## 現在地と次の一手
1. **生成が未完了**。Qwen3-8B の重みは取得済みだが、vLLM 起動時に FlashInfer
   サンプラーが sm120 で失敗 (`FlashInfer requires GPUs with sm75 or higher`)。
   回避案: `VLLM_USE_FLASHINFER_SAMPLER=0` を付けて再実行 (未検証)。
   代替: vLLM をやめて Transformers で生成。
2. 100問×3案を `data/cands_base.jsonl` に出力 (ベースライン)。
3. `arena.py` で人間がA/B投票 → `prefs.jsonl` / `battles.jsonl` が貯まる。
   (arena は1お題に2モデル以上の回答が必要。ベース+別モデルの回答が要る)
4. 投票から DPO → 再生成 → 再評価のループ。

## 注意・リスク
- お題が少ない(学習90問)ため過学習しやすい。データ拡張と早期打ち切りに注意
- Judge LLM は既視感・長文に甘い偏りがある。人間投票との相関を定期確認
- 大喜利サイトの無断スクレイピングは避ける。自前/許諾データのみ
- ディスク逼迫: チェックポイントや不要なキャッシュは随時削除

## 更新 (SFT実施後)
- 公開データ `iammytoo/japanese-humor-evaluation-v2` のテキストお題・高得点(score>=3.5) 4,407件/929お題で QLoRA SFT (2 epoch, 約11分, loss 2.36→2.06)
- 生成温度1.1では日本語が崩れる。**温度0.7で崩れが大幅に減り、人間(sinchir0)の主観でもSFTがベースより良い**と判断
- 基準モデル: Qwen3-8B + ckpt/sft, temperature 0.7
- 次: ベース/SFTをA/B比較する arena で人間投票 → 選好データ → DPO
- 注意: 元データはNHK番組由来で権利未確認。研究目的に限る

## 更新 (投票結果と現在地) 2026-09-29
### A/B投票 (温度0.7, 1お題3案, 35票) — data/old/battles_v1.jsonl
- 引き分け(どちらも微妙) 25票(71%)、勝敗がついた10票は base 7勝 / SFT 3勝。Elo は base≈1020, SFT≈980 で有意差なし
- 先の「SFT の方が良い」という判断は、2問の印象でしかなく、この集計とは合わなかった

### 多様生成 + 6案から1つ選ぶ (pick-best, 13セット) — data/picks.jsonl
- 13セット中9セット(69%)が「全部微妙」。選ばれたのは 4 件のみ (誇張3, 具体的1)
- 選択率(ランダム=16.7%): base 4.5%(2/44), sft 5.9%(2/34)。件数が少なくモデル間の差は判断不能
- 結論: **現状のモデル(Qwen3-8B, SFT含む)の出力は、サンプル数を増やしても面白くない**
  - 温度を下げると無難になり、上げると日本語が崩れる。多様生成でも当たりが出ない

### 主な仮説と次の候補
1. 8B の素の能力・SFTデータ(ケータイ大喜利の文脈依存回答)の限界
2. 強い外部モデル(Claude 等)で候補生成/Judge/蒸留 → 要 API キー・外部送信の可否決定
3. Judge で 32〜64 案から上位を絞る Best-of-N は未検証 (Judge 自体が 8B なので過信不可)
4. DPO(公開ペア806組 + 人間ペア18組)は根本解決にならない可能性が高い

### 状態
- 未push: SFT の LoRA (ckpt/sft, 約170MB級, git 管理外)。永続化するなら HF Hub の private repo 推奨
- アリーナは nohup 起動(再起動で停止): `python -m ogiri.arena --port 17080`

## 更新 (テキスト+画像のVLM化、Qwen3.5移行) 2026-10-03

### 方針転換
大喜利を「テキストのみのお題」と「画像(+テキスト)のお題」の両方に対応させる。
別モデルに分けず、**1つのVLMに両方を学習させる**方針(Generator/Judgeとも同一モデルで
画像を見られるようにする)。

### ベースモデル変更: Qwen3-8B → Qwen3.5-9B
- Qwen3.5(2026-03公開)はネイティブマルチモーダル(テキスト/画像/動画を1モデルで処理)。
  「Qwen3.5-8B」という名称は存在せず、実在するのは **Qwen3.5-9B**(Smallシリーズ: 0.8B/2B/4B/9B)。
- アーキテクチャはGated DeltaNet(線形Attention)とフルAttentionの3:1ハイブリッド。
  高速カーネル(`causal_conv1d`/`flash-linear-attention`)はBlackwell(sm120)向けビルドが無く、
  **PyTorchフォールバック(低速)で動作**。torch/vLLM/transformersのビルド済みwheelはcu128以降が必要。
- 公式に **QLoRA(4bit)は非推奨**(量子化誤差が通常より大きい)。`sft.py` は bf16 LoRA に変更
  (target_modules は `q/k/v/o_proj, gate/up/down_proj` に限定。DeltaNet特有の線形層やvision towerはLoRA対象外)。
- 実測スループット(9B, フォールバックカーネル, RTX PRO 5000 Blackwell 48GB): 画像あり 2.76 samples/s、
  テキストのみ 7.7程度(4Bの実測から推定)。バッチサイズを上げても速度は変わらない(compute-bound)。

### 画像データセット: zhongshsh/CLoT-Oogiri-GO (日本語)
- Bokete由来、CC BY 4.0。T2T 11,842 / I2T 40,278 / IT2T 9,420件(日本語)。
- **重要**: 日本語データはT2T(本来テキストのみの型)も含めて**全件に画像が紐づいている**
  (お題文が画像として焼き込まれている)。純粋なテキストのみのお題は含まれない。
  → 埋め込み文字の読み取りはOCRせず、VLM自身の視覚的文字認識に委ねる方針。
- `ogiri/prepare_data_clot.py` (新規) で画像(14,830枚、522MB)を `data/images/clot/` に展開し、
  star(いいね数)を同一画像内パーセンタイルでwin/ok/bad化 → `data/sft_clot.jsonl`(61,538件)
  / `data/prefs_clot.jsonl`(31,088件選好ペア)を生成。
- テキストのみのSFTデータは従来通り `iammytoo/japanese-humor-evaluation-v2` (`prepare_data.py`)
  から生成 (`data/sft.jsonl`, 4,407件/929お題)。

### common.py / benchmark.py の変更
- `chat()`: 全role(system/user/assistant)のcontentを常にblock形式(`[{"type":"text",...}]`等)に統一。
  理由: HF `Dataset.from_list` がroleごとにcontent型(str/list)が割れるとpyarrowで落ちる。
- `topic_hash()` / `filter_train()`: 画像お題(`image`フィールド)もハッシュ対象にして
  将来の画像ベンチマークでもリーク防止できるようにした。

### sft.py 全面書き換え
- `AutoProcessor` + カスタム `collate_fn` でテキスト/画像混在データセットを学習。
- プロンプト部分の長さを `add_generation_prompt=True` で再計算し、labelsをマスク
  (回答部分のみに損失をかける、という元の設計を維持)。
- `mm_token_type_ids` をprocessorから取得してモデルに渡す必要がある(Qwen3.5の3D RoPE計算に必須)。
- 生成時は `enable_thinking=False` を渡さないと英語のThinking Processを書き始めて
  短い答えに辿り着かない(既存の`generate.py`/`judge.py`は元々対応済み)。

### 動作確認(100件トライアル学習)
- テキスト50件+画像50件、3epochで学習 → loss 3.42→2.17、token精度0.45→0.67と収束。
- 学習後、ベースモデル+LoRAで生成テスト。テキストのみ/画像のみ/画像+テキストの3パターンとも、
  「前置きなしの短い日本語」という学習した出力フォーマットを正しく再現できることを確認。
- チェックポイントは `ckpt/trial100` としてリポジトリ内に保存(git管理外、ckpt/はgitignore対象)。

### デモUI (`ogiri/demo.py`, 新規)
- テキストのみ/画像のみ/画像+テキストの3パターンを1画面で試せるブラウザUI。
- supervisorサービス化(`ogiri-demo`)、Caddy経由ポート10200で外部公開(トークン認証)。

### 本番SFT実行とトラブル
- `data/sft.jsonl`(4,407件) + `data/sft_clot.jsonl`のwin tier(22,860件)、計27,267件・1epochで実行。
- **1回目: 763/853ステップ(89%、約2時間経過)でクラッシュ**。
  CLoT画像データの中に1枚だけ異常なアスペクト比(600x1px)の画像があり、
  Qwen系image processorの上限(アスペクト比200)に引っかかってエラー終了。
  `save_strategy="epoch"`だったため、2時間分の学習が保存されず消失。
- 対策: `prepare_data_clot.py`にアスペクト比フィルタ(閾値150)を追加、
  `sft.py`の`collate_fn`は個別の壊れた例をスキップしてバッチ全体を落とさないよう修正。
  問題の画像1枚と関連行はデータから除去済み。
- 2回目を再実行(GPUメモリ確保のため`ogiri-demo`サービスは学習中一時停止)。

### 運用方針
- GitHubへのpushは常に実施(private repo `sinchir0/ogiri-llm`)。トークンはコミット対象に含めない
  (`/workspace/.env` はリポジトリ外)。
- 学習済み重みはHugging Face Hubにアップロード予定: `sinchir0/ogiri-qwen3.5-9b-sft` (private)。
- 学習・アップロード・push完了後、Vastインスタンスは**確認なしでdestroy**してよいと指示あり
  (`/workspace`はvolumeだが、GitHub/HFを正規の永続化先として扱う)。

### 次の一手
1. 本番SFT完了 → HF Hubへアップロード → GitHubへpush → インスタンスdestroy
2. DPO(`dpo.py`)もマルチモーダル対応が必要(現状`chat()`にimageを渡していない、未着手)
3. Judgeのマルチモーダル化(画像お題の「的確に絡んでいるか」を判定するには画像を見せる必要がある)
4. Arenaで画像お題の人間投票も収集(`arena.py`は現状テキストのみ想定のUI、画像表示は未対応)
5. 本番チェックポイントでの生成品質評価(temperature調整、Elo比較)

## 更新 (本番SFT完了・HFアップロード) 2026-10-03

- 1回目のクラッシュ(前セクション参照)修正後、再実行して **853/853ステップ(1 epoch)完走**。
  所要時間 約2時間14分。train_loss 3.09、mean_token_accuracy 0.45→0.50。
- データ: テキスト(iammytoo) 4,407件 + 画像(CLoT-Oogiri-GO ja, win tier) 22,860件 = 27,267件、1epoch。
- 生成確認(テキストのみ/画像のみ)で、短い日本語の直接回答が出ることを確認済み。
- 重みを Hugging Face Hub にアップロード: [`sinchir0/ogiri-qwen3.5-9b-sft`](https://huggingface.co/sinchir0/ogiri-qwen3.5-9b-sft) (private)。
  モデルカードに学習データ・使い方・ライセンス注意事項を記載。
- `ogiri-demo`サービス(学習中は停止していた)は再起動せず、インスタンスをこのあとdestroyする前提で終了。
- 残課題(DPOのマルチモーダル対応、Judgeのマルチモーダル化、Arenaの画像対応、品質評価)は上記「次の一手」のまま未着手。次回インスタンスを立てたら、このHF重みとGitHubのコードから再開できる。

## 更新 (手順SFT → DPO、デモUI) 2026-10-04
- 手順プロンプトの条件比較 (A〜D) で、画像系は手順プロンプト(B)が強く、テキストは差が小さいことを確認。
  「1モデルで全入力」の要件から、手順を SFT で内在化する方針に変更。
- 思考データは、人間に高評価された回答から逆算してベースモデルに5行で書かせた (7,251件有効)。
- 手順SFT: 227ステップ・約37分 (flash-linear-attention 導入)。DPO は自作 `ogiri/dpo_proc.py` (思考+回答、画像対応)。
- 人間投票 105件 (全部微妙 25%)、選好ペア230組で DPO。学習は過学習、検証 61〜67% (33ペア) で有意差なし。
- 詳細は [TRAINING.md](TRAINING.md)。デモUI: `ogiri/demo.py` (supervisor `ogiri-demo`, 外部ポート10100)。
- 環境メモ: 新しい環境では `datasets`/`trl` が古い版で入るため `uv pip install "datasets>=4" "trl>=1.0"` が必要。
  ディスクが逼迫しやすい (32GB) ので、途中保存のチェックポイントは学習後に削除する。
