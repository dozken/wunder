#!/bin/bash
# Resumable multi-segment download of the starter pack, then streaming extract.
# Each segment is appended to its own part file via HTTP Range so any
# interruption resumes from the bytes already on disk.
set -u
URL="https://files.wundernn.io/wnn_connectome_starterpack.tar.gz"
SIZE=33737957587
N=6
DIR="$(cd "$(dirname "$0")/../.dl" 2>/dev/null && pwd || { mkdir -p "$(dirname "$0")/../.dl" && cd "$(dirname "$0")/../.dl" && pwd; })"
DEST="$(cd "$DIR/.." && pwd)"       # challenge-3/
cd "$DIR"

CHUNK=$(( (SIZE + N - 1) / N ))

fetch_part() {
  local i=$1
  local start=$(( i * CHUNK ))
  local end=$(( (i + 1) * CHUNK - 1 ))
  [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
  local want=$(( end - start + 1 ))
  local f="part.$i"
  [ -f "$f" ] || : > "$f"
  while :; do
    local have
    have=$(stat -f %z "$f")
    if [ "$have" -ge "$want" ]; then
      echo "[part $i] complete ($have bytes)"
      return 0
    fi
    local from=$(( start + have ))
    echo "[part $i] resuming at offset $have / $want"
    curl -sS --fail -m 0 --speed-limit 10000 --speed-time 60 \
      -r "$from-$end" "$URL" >> "$f" || echo "[part $i] curl exit $? -- retrying in 10s"
    sleep 10
  done
}

echo "start $(date -u +%FT%TZ)"
for i in $(seq 0 $((N - 1))); do fetch_part "$i" & done
wait
total=0
for i in $(seq 0 $((N - 1))); do total=$(( total + $(stat -f %z "part.$i") )); done
echo "downloaded $total bytes (expected $SIZE) $(date -u +%FT%TZ)"
[ "$total" -eq "$SIZE" ] || { echo "SIZE MISMATCH"; exit 1; }

echo "extracting $(date -u +%FT%TZ)"
cat $(for i in $(seq 0 $((N - 1))); do echo "part.$i"; done) | tar -xz -C "$DEST" || { echo "EXTRACT FAILED"; exit 1; }
echo "extracted $(date -u +%FT%TZ)"
rm -f part.*
ls -la "$DEST/wnn_connectome_starterpack/datasets"
echo "done $(date -u +%FT%TZ)"
