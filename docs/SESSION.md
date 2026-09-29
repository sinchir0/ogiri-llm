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
