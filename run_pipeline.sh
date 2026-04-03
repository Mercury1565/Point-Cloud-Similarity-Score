#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full Point Cloud Similarity Score pipeline
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --datasets      Comma-separated list of datasets to process.
#                   Choices: nuscenes, nuscenes_full, kitti, waymo, all   (default: nuscenes,kitti)
#                   nuscenes_full uses the interpolated non-keyframe pkl annotations (~20 Hz)
#   --skip-extract  Skip raw data extraction (use existing unified_*.json files)
#   --skip-csv      Skip CSV generation (use existing data/csv/*.csv files)
#   --skip-static   Deprecated no-op (static model stage removed; Bayesian-only pipeline)
#   --online-only   Equivalent to --skip-extract --skip-csv
#   --model-dir     Directory to save/load Bayesian model states (default: models)
#   --output-dir    Directory for plots (default: output)
#   --audit         Run data audit after extraction (nuScenes only)
#   --threshold     Confidence threshold for online engine (default: 0.85)
#   --threshold-map Per-dataset threshold overrides: "dataset:value,..."
#                   Example: "nuscenes:0.9,nuscenes_full:0.8,kitti:0.7,waymo:0.4"
#   --audit-interval   Frames between Bayesian updates (default: 5)
#   --seed-fraction Fraction of dataset for cold-start training (default: 0.15)
#   --split-mode    Seed split strategy: frame or scene (default: frame)
#   --alpha         Prior precision (default: 1.0)
#   --beta          Noise precision (default: 25.0)
#   --uncertainty-weight  k in mean-k*std rule (default: 1.0)
#   -h, --help      Show this message
#
# Examples:
#   # Full pipeline, nuScenes + KITTI (default)
#   ./run_pipeline.sh
#
#   # Full pipeline including Waymo
#   ./run_pipeline.sh --datasets nuscenes,kitti,waymo
#
#   # Re-run only the online engine (CSVs already exist)
#   ./run_pipeline.sh --online-only
#
#   # Single dataset, save trained model
#   ./run_pipeline.sh --datasets kitti --model-dir models/
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DATASETS="nuscenes,kitti"
SKIP_EXTRACT=false
SKIP_CSV=false
RUN_AUDIT=false
MODEL_DIR="models"
OUTPUT_DIR="output"
THRESHOLD=0.85
THRESHOLD_MAP=""
AUDIT_INTERVAL=5
SEED_FRACTION=0.15
SPLIT_MODE="frame"
ALPHA=1.0
BETA=25.0
UNCERTAINTY_WEIGHT=1.0

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)             DATASETS="$2";            shift 2 ;;
    --skip-extract)         SKIP_EXTRACT=true;        shift   ;;
    --skip-csv)             SKIP_CSV=true;            shift   ;;
    --skip-static)
      echo "[pipeline] --skip-static is deprecated: static model stage has been removed (Bayesian-only)."
      shift ;;
    --online-only)          SKIP_EXTRACT=true; SKIP_CSV=true; shift ;;
    --audit)                RUN_AUDIT=true;           shift   ;;
    --model-dir)            MODEL_DIR="$2";           shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2";          shift 2 ;;
    --threshold)            THRESHOLD="$2";           shift 2 ;;
    --threshold-map)        THRESHOLD_MAP="$2";       shift 2 ;;
    --audit-interval)       AUDIT_INTERVAL="$2";      shift 2 ;;
    --seed-fraction)        SEED_FRACTION="$2";       shift 2 ;;
    --split-mode)           SPLIT_MODE="$2";          shift 2 ;;
    --alpha)                ALPHA="$2";               shift 2 ;;
    --beta)                 BETA="$2";                shift 2 ;;
    --uncertainty-weight)   UNCERTAINTY_WEIGHT="$2";  shift 2 ;;
    -h|--help)
      sed -n '/^# Usage:/,/^# =====/{/^# =====/!p}' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BOLD}[pipeline]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[skip]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# Resolve the project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick an available Python executable once and reuse it.
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  die "Neither 'python' nor 'python3' was found on PATH"
fi

# Expand "all" to every known dataset
if [[ "$DATASETS" == "all" ]]; then
  DATASETS="nuscenes,nuscenes_full,kitti,waymo"
fi

if [[ "$SPLIT_MODE" != "frame" && "$SPLIT_MODE" != "scene" ]]; then
  die "--split-mode must be 'frame' or 'scene' (got '$SPLIT_MODE')"
fi

# Split comma-separated list into an array
IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

# Parse optional per-dataset threshold overrides
declare -A THRESHOLD_BY_DATASET
if [[ -n "$THRESHOLD_MAP" ]]; then
  IFS=',' read -ra threshold_pairs <<< "$THRESHOLD_MAP"
  for pair in "${threshold_pairs[@]}"; do
    [[ -z "$pair" ]] && continue

    if [[ "$pair" != *:* ]]; then
      die "Invalid --threshold-map entry '$pair' (expected dataset:value)"
    fi

    ds="${pair%%:*}"
    thr="${pair#*:}"

    case "$ds" in
      nuscenes|nuscenes_full|kitti|waymo) ;;
      *) die "Invalid dataset '$ds' in --threshold-map (allowed: nuscenes,nuscenes_full,kitti,waymo)" ;;
    esac

    if [[ ! "$thr" =~ ^([0-9]*[.])?[0-9]+$ ]]; then
      die "Invalid threshold '$thr' for dataset '$ds' in --threshold-map"
    fi

    if ! awk -v v="$thr" 'BEGIN { exit !(v >= 0 && v <= 1) }'; then
      die "Threshold for dataset '$ds' must be within [0,1] (got '$thr')"
    fi

    THRESHOLD_BY_DATASET["$ds"]="$thr"
  done
fi

log "Datasets : ${DATASET_LIST[*]}"
log "Output   : $OUTPUT_DIR"
log "Models   : $MODEL_DIR"
log "Split    : $SPLIT_MODE"
log "Threshold: $THRESHOLD"
if [[ -n "$THRESHOLD_MAP" ]]; then
  log "Overrides: $THRESHOLD_MAP"
fi
echo ""

# --------------------------------------------------------------------------- #
# Step 1 — Extract raw data → Unified JSON
# --------------------------------------------------------------------------- #
if $SKIP_EXTRACT; then
  warn "Skipping extraction (--skip-extract)"
else
  log "=== STEP 1: Data Extraction ==="
  for ds in "${DATASET_LIST[@]}"; do
    script="extract/extract_${ds}.py"
    if [[ ! -f "$script" ]]; then
      warn "No extraction script found for '$ds' ($script) — skipping"
      continue
    fi
    log "Extracting $ds..."
    "$PYTHON_BIN" "$script" || die "Extraction failed for $ds"
    ok "$ds extraction complete"
  done
fi
echo ""

# --------------------------------------------------------------------------- #
# Step 2 — Audit (optional, nuScenes only)
# --------------------------------------------------------------------------- #
if $RUN_AUDIT; then
  if [[ " ${DATASET_LIST[*]} " == *" nuscenes "* ]]; then
    log "=== STEP 2: Data Audit (nuScenes) ==="
    "$PYTHON_BIN" extract/audit_nuscenes.py || die "Audit failed"
    ok "Audit complete — see nuscenes_similarity_trends.png"
  else
    warn "Audit requested but nuScenes not in dataset list — skipping"
  fi
fi
echo ""

# --------------------------------------------------------------------------- #
# Step 3 — Generate training CSVs
# --------------------------------------------------------------------------- #
if $SKIP_CSV; then
  warn "Skipping CSV generation (--skip-csv)"
else
  log "=== STEP 3: CSV Generation ==="
  for ds in "${DATASET_LIST[@]}"; do
    script="extract/generate_csv_${ds}.py"
    if [[ ! -f "$script" ]]; then
      warn "No CSV script found for '$ds' ($script) — skipping"
      continue
    fi
    # Check that the unified JSON exists before attempting CSV generation
    json_map_nuscenes="unified_nuscenes_mini.json"
    json_map_nuscenes_full="unified_nuscenes_full.json"
    json_map_kitti="unified_kitti.json"
    json_map_waymo="unified_waymo.json"
    json_var="json_map_${ds}"
    json_file="${!json_var}"
    if [[ ! -f "$json_file" ]]; then
      warn "Unified JSON not found ($json_file) — run extraction first or use --skip-extract"
      continue
    fi
    log "Generating CSV for $ds..."
    "$PYTHON_BIN" "$script" || die "CSV generation failed for $ds"
    ok "$ds CSV complete"
  done
fi
echo ""

# --------------------------------------------------------------------------- #
# Step 4 — Online Bayesian engine (one model per dataset)
# --------------------------------------------------------------------------- #
log "=== STEP 4: Online Bayesian Engine ==="

# Build --datasets arg: only include datasets that actually have a CSV
available_ds=()
for ds in "${DATASET_LIST[@]}"; do
  csv_count=$(find data/csv -name "*${ds}*.csv" 2>/dev/null | wc -l)
  if [[ "$csv_count" -gt 0 ]]; then
    available_ds+=("$ds")
  else
    warn "No CSV found for '$ds' in data/csv/ — skipping online engine for this dataset"
  fi
done

if [[ ${#available_ds[@]} -eq 0 ]]; then
  warn "No CSVs available for any requested dataset — skipping online engine"
else
  mkdir -p "$OUTPUT_DIR" "$MODEL_DIR"

  for ds in "${available_ds[@]}"; do
    ds_threshold="$THRESHOLD"
    if [[ -n "${THRESHOLD_BY_DATASET[$ds]+x}" ]]; then
      ds_threshold="${THRESHOLD_BY_DATASET[$ds]}"
    fi

    log "Running online engine for $ds (threshold=$ds_threshold)..."
    "$PYTHON_BIN" -m online_model \
      --dataset "$ds" \
      --csv-dir data/csv \
      --output-dir "$OUTPUT_DIR" \
      --model-state-dir "$MODEL_DIR" \
      --confidence-threshold "$ds_threshold" \
      --audit-interval "$AUDIT_INTERVAL" \
      --seed-fraction "$SEED_FRACTION" \
      --split-mode "$SPLIT_MODE" \
      --alpha "$ALPHA" \
      --beta "$BETA" \
      --uncertainty-weight "$UNCERTAINTY_WEIGHT" \
      || die "Online engine failed for $ds"
    ok "$ds online engine complete"
  done
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Pipeline complete${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo "  Plots       : $OUTPUT_DIR/{$(IFS=,; echo "${available_ds[*]+"${available_ds[*]}"}")}"
echo "  Model states: $MODEL_DIR/*.npz"
echo ""
