#!/usr/bin/env bash
# Disk-aware fetch of the locked Xenium technical-replicate noise-floor source (Janesick 2023 breast).
#   Rep1 = GSM7780153, Rep2 = GSM7780154 (same FFPE block, serial 5um sections, same 313-gene panel).
#
# The cell-by-gene matrix lives INSIDE each ~9.2 GB outs.zip, so we stream-extract-delete: download the
# zip, pull ONLY {cell_feature_matrix.h5, cells.parquet|cells.csv.gz, gene_panel.json} out of it, then
# delete the zip. Peak disk stays ~9.2 GB (one zip at a time) -- fits a tight (~22 GB-free) disk.
#
# Usage:
#   bash scripts/fetch_data.sh [OUT_DIR] [MODE]
#     OUT_DIR  default: data
#     MODE     noise  (default) -> just the two Xenium matrices + panel  (enough for the sec-6 control)
#              full              -> also the 1.3 GB post-Xenium H&E + homography per replicate (for U)
set -euo pipefail

OUT="${1:-data}"
MODE="${2:-noise}"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7780nnn"
PY="${PYTHON:-python3}"

# NB: no associative arrays (bash 3.2, the macOS default, lacks `declare -A`) -- audit SB2.
gsm_of() { case "$1" in rep1) echo GSM7780153;; rep2) echo GSM7780154;; esac; }
tag_of() { case "$1" in rep1) echo Rep1;; rep2) echo Rep2;; esac; }

# portable free-GB (df -g is macOS-only): POSIX `df -Pk` gives 1024-byte blocks on macOS AND Linux.
free_gb() { df -Pk "$OUT" | awk 'NR==2{print int($4/1024/1024)}'; }

fetch() {  # fetch <url> <out_path>  (resumable)
  echo "  -> $(basename "$2")"
  curl -fL -C - --retry 5 --retry-delay 5 -o "$2" "$1"
}

mkdir -p "$OUT"
echo "free disk on $OUT: $(free_gb) GB"

for rep in rep1 rep2; do
  g="$(gsm_of "$rep")"; tag="$(tag_of "$rep")"
  dir="$OUT/$rep"; mkdir -p "$dir"
  suppl="$BASE/$g/suppl"
  prefix="${g}_Xenium_FFPE_Human_Breast_Cancer_${tag}"

  echo "=== $rep ($g) ==="
  # 1) standalone panel (tiny) for scripts/check_panel.py
  fetch "$suppl/${prefix}_gene_panel.json.gz" "$dir/${prefix}_gene_panel.json.gz" || true
  if [ -f "$dir/${prefix}_gene_panel.json.gz" ]; then gunzip -kf "$dir/${prefix}_gene_panel.json.gz"; fi

  # 2) the 9.2 GB outs.zip -> selective extract -> delete
  if [ "$(free_gb)" -lt 11 ]; then
    echo "  !! < 11 GB free; the 9.2 GB outs.zip may not fit. Free space and rerun." >&2; exit 1
  fi
  zip="$dir/${prefix}_outs.zip"
  fetch "$suppl/${prefix}_outs.zip" "$zip"
  echo "  extracting matrix + cells + panel from $(basename "$zip") ..."
  PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" "$PY" -c "
import sys
from src import xenium_io as x
got = x.extract_members(sys.argv[1], sys.argv[2],
    ('cell_feature_matrix.h5','cells.parquet','cells.csv.gz','gene_panel.json'))
print('   extracted:', [p.name for p in got])
" "$zip" "$dir"
  rm -f "$zip"
  echo "  deleted $(basename "$zip"); free disk now: $(free_gb) GB"

  # 3) full mode: post-Xenium H&E (1.3 GB) + homography, for morphology embeddings
  if [ "$MODE" = "full" ]; then
    fetch "$BASE/$g/suppl/${g}_Post-Xenium_HE_${tag}.ome.tif.gz" "$dir/${g}_Post-Xenium_HE_${tag}.ome.tif.gz" || true
    fetch "$BASE/$g/suppl/${g}_Post-Xenium_HE_${tag}_homography.csv.gz" "$dir/${g}_Post-Xenium_HE_${tag}_homography.csv.gz" || true
  fi
done

echo
echo "done. next:"
echo "  python3 scripts/check_panel.py $OUT/rep1 $OUT/rep2          # confirm identical panel"
echo "  python3 experiments/noise_floor_breast.py $OUT             # the sec-6 noise-floor control (no UNI)"
if [ "$MODE" != "full" ]; then
  echo "  (re-run with MODE=full to also fetch H&E for the morphology/U gate: bash scripts/fetch_data.sh $OUT full)"
fi
