#!/bin/bash
# 新しい Vast.ai インスタンス(Vast ベースイメージ)で、大喜利デモUIを立ち上げる。何度実行しても壊れない(冪等)。
#
#   git clone https://github.com/sinchir0/ogiri-llm.git && cd ogiri-llm && bash deploy/setup_demo.sh
#
# やること: 依存のインストール → ベースモデルと LoRA(HF)の取得 → supervisor サービス登録
#           → portal(Caddy 認証つき外部公開)登録 → 起動待ち → 3種類の入力でスモークテスト → URL 表示
# 環境変数(任意): HF_REPO(LoRAのHFリポジトリ) / BASE_MODEL / EXTERNAL_PORT(公開ポートを固定) / HF_TOKEN(privateの場合)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO="${HF_REPO:-sinchir0/ogiri-qwen3.5-9b-proc}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
VENV="${VENV:-/venv/main}"
PY="$VENV/bin/python"   # /etc/environment 等で PATH が変わっても動くよう絶対パスで呼ぶ
SVC=ogiri-demo
LABEL="Ogiri Demo"
INTERNAL_PORT=17090
WAIT_SEC="${WAIT_SEC:-1500}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
# 環境変数の値を、現在の環境 → /etc/environment の順で取り出す(/etc/environment を丸ごと読むと PATH が上書きされるため)
get_env() { local v="${!1:-}"; [ -n "$v" ] || v=$(grep -E "^$1=" /etc/environment 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'); printf %s "$v"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "0/7 前提の確認"
command -v nvidia-smi >/dev/null || die "GPU が見つかりません (nvidia-smi)"
command -v supervisorctl >/dev/null || die "supervisorctl がありません。Vast のベースイメージで実行してください"
[ -x "$VENV/bin/python" ] || die "$VENV がありません。VENV=<venv のパス> で指定してください"
FREE_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 45 ] || echo "WARN: 空き容量 ${FREE_GB}GB。ベースモデル(約19GB)+ 依存(約10GB)に 45GB 以上を推奨します"
echo "repo=$REPO / venv=$VENV / HF=$HF_REPO / base=$BASE_MODEL / 空き ${FREE_GB}GB"

say "1/7 依存のインストール (requirements.txt)"
# shellcheck disable=SC1091
set +u; . "$VENV/bin/activate"; set -u
if command -v uv >/dev/null; then
  UV_HTTP_TIMEOUT=300 uv pip install --no-cache -r "$REPO/requirements.txt"
else
  pip install -r "$REPO/requirements.txt"
fi

say "2/7 モデルの取得 (ベース + LoRA)"
HF_REPO="$HF_REPO" BASE_MODEL="$BASE_MODEL" REPO="$REPO" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
base = snapshot_download(os.environ["BASE_MODEL"])
print("base :", base)
lora = snapshot_download(os.environ["HF_REPO"], allow_patterns=["adapter_*", "sft_proc/adapter_*"],
                         local_dir=os.path.join(os.environ["REPO"], "ckpt", "hf"))
print("lora :", lora)
for f in ("adapter_model.safetensors", "sft_proc/adapter_model.safetensors"):
    p = os.path.join(lora, f)
    assert os.path.getsize(p) > 1e8, f"{p} が小さすぎます"
PY

say "3/7 supervisor サービスの登録"
cat > "/opt/supervisor-scripts/$SVC.sh" <<EOF
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
. "\${utils}/exit_portal.sh" "$LABEL"

source $VENV/bin/activate
cd $REPO
export VLLM_USE_FLASHINFER_SAMPLER=0
pty python -m ogiri.demo --lora sft=ckpt/hf/sft_proc --lora dpo=ckpt/hf --port $INTERNAL_PORT 2>&1
EOF
chmod +x "/opt/supervisor-scripts/$SVC.sh"
cat > "/etc/supervisor/conf.d/$SVC.conf" <<EOF
[program:$SVC]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/$SVC.sh
autostart=true
autorestart=unexpected
startsecs=5
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
EOF

say "4/7 portal(認証つき外部公開)の登録"
EXT=$(LABEL="$LABEL" INTERNAL_PORT="$INTERNAL_PORT" EXTERNAL_PORT="${EXTERNAL_PORT:-}" "$PY" - <<'PY'
import json, os, subprocess, yaml
label, internal = os.environ["LABEL"], int(os.environ["INTERNAL_PORT"])
d = yaml.safe_load(open("/etc/portal.yaml")) or {"applications": {}}
apps = d.setdefault("applications", {})
ext = os.environ.get("EXTERNAL_PORT") or (apps.get(label) or {}).get("external_port")  # 指定 > 既存の登録を再利用
if not ext:
    caps = json.loads(subprocess.run(["vast-capabilities"], capture_output=True, text=True).stdout.split("\n>>>")[0])
    free = [p["container_port"] for p in caps["instance"]["open_ports"]
            if not p["in_use"] and p["container_port"] <= 65535 and p["proto"] == "tcp"]
    ext = free[0] if free else ""
if ext:
    apps[label] = {"hostname": "localhost", "external_port": int(ext), "internal_port": internal,
                   "open_path": "/", "name": label}
    yaml.safe_dump(d, open("/etc/portal.yaml", "w"), sort_keys=False)
print(ext)
PY
)
if [ -z "$EXT" ]; then
  # exit_portal.sh は portal.yaml に登録が無いとサービスが起動しないため、内部ポートだけ登録して外部公開はしない
  "$PY" - <<PY
import yaml
d = yaml.safe_load(open("/etc/portal.yaml")) or {"applications": {}}
d.setdefault("applications", {})["$LABEL"] = {"hostname": "localhost", "external_port": $INTERNAL_PORT, "internal_port": $INTERNAL_PORT, "open_path": "/", "name": "$LABEL"}
yaml.safe_dump(d, open("/etc/portal.yaml", "w"), sort_keys=False)
PY
  echo "WARN: 空いている外部ポートがありません。SSH ポートフォワードで使ってください:"
  echo "  ssh -p \$VAST_TCP_PORT_22 -L 8080:127.0.0.1:$INTERNAL_PORT root@\$PUBLIC_IPADDR  → http://localhost:8080"
else
  echo "外部ポート: $EXT -> 内部 $INTERNAL_PORT"
fi
supervisorctl restart caddy >/dev/null || true

say "5/7 サービスの起動"
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
supervisorctl restart "$SVC" >/dev/null 2>&1 || supervisorctl start "$SVC" >/dev/null
supervisorctl status "$SVC"

say "6/7 起動待ち (初回は torch.compile / モデル読み込みで 5〜10 分かかります)"
ELAPSED=0
until curl -s -o /dev/null "http://127.0.0.1:$INTERNAL_PORT/api/models"; do
  sleep 5; ELAPSED=$((ELAPSED + 5))
  supervisorctl status "$SVC" | grep -q RUNNING || die "サービスが停止しました。ログ: /var/log/portal/$SVC.log"
  [ "$ELAPSED" -lt "$WAIT_SEC" ] || die "${WAIT_SEC}秒待っても起動しません。ログ: /var/log/portal/$SVC.log"
  [ $((ELAPSED % 60)) -ne 0 ] || echo "  ... ${ELAPSED}s"
done
echo "起動しました (${ELAPSED}s)。モデル: $(curl -s "http://127.0.0.1:$INTERNAL_PORT/api/models")"

say "7/7 スモークテスト (テキスト / 画像 / 画像+テキスト)"
TMPIMG="$(mktemp --suffix=.png)"
"$PY" - "$TMPIMG" <<'PY'
import sys
from PIL import Image, ImageDraw
im = Image.new("RGB", (448, 336), (230, 220, 200)); d = ImageDraw.Draw(im)
d.ellipse((120, 60, 330, 270), fill=(200, 60, 60)); d.rectangle((40, 250, 400, 300), fill=(60, 90, 160))
im.save(sys.argv[1])
PY
check() {  # $1=名前 以降=curl の引数
  local name="$1"; shift
  local out; out=$(curl -s -m 180 -X POST "http://127.0.0.1:$INTERNAL_PORT/api/generate" "$@" -F model=dpo -F n=2)
  "$PY" - "$name" "$out" <<'PY'
import json, sys
name, raw = sys.argv[1], sys.argv[2]
try:
    j = json.loads(raw)
except Exception:
    sys.exit(f"  NG  {name}: JSON ではない応答: {raw[:200]}")
if "error" in j:
    sys.exit(f"  NG  {name}: {j['error']}")
print(f"  OK  {name}: {[c['answer'] for c in j['candidates']]}")
PY
}
check "テキストのみ" -F topic="こんなコンビニは嫌だ。どんなコンビニ？"
check "画像のみ" -F image=@"$TMPIMG"
check "画像+テキスト" -F image=@"$TMPIMG" -F topic="この写真に写っている物の本当の正体とは？"
rm -f "$TMPIMG"

say "完了"
if [ -n "$EXT" ]; then
  echo "URL: http://$(get_env PUBLIC_IPADDR):$(get_env "VAST_TCP_PORT_$EXT")/?token=$(get_env OPEN_BUTTON_TOKEN)"
fi
