#!/bin/bash

# GPU-aware wrapper for SLURM orchestration.
# Resolves SLURM parameters from a GPU mapping table and passes the full
# node list to SLURM, letting its scheduler pick the least-loaded node
# for each job.

set -euo pipefail

# --- Usage ---
usage() {
    cat <<'EOF'
Usage:
  ./run_benchmark_gpu.sh --gpu <type> --config <file> [options]
  ./run_benchmark_gpu.sh --gpu <type> --resume <dir>  [options]

Required flags:
  --gpu <type>          GPU type: V100, A100_SXM, A100_PCIe, H100_SXM, H100_PCIe, H200, L40S
  --config <file>       Main YAML config file (mutually exclusive with --resume)

Optional flags:
  --resume <dir>        Resume from existing temp dir (mutually exclusive with --config)
  --account <name>      SLURM account (default: init)
  --batch-size <n>      Models per batch config (default: 1)
  --max-jobs <n>        Max concurrent SLURM jobs (default: 500)
  --nodes <list>        Comma-separated node subset override
  --max-retries <n>     Max GPU mismatch retries for H100 variants (default: 10)
  --exclusive           Request exclusive node access (jobs run sequentially, safer for shared nodes)
  --dry-run             Pass through to run_all_slurm.sh
  -h, --help            Show this help message

GPU types and their defaults:
  V100       partition=gpu,     time=360, gres=gpu:v100sxm:1,  nodes: losangeles,sanfrancisco,sandiego
  A100_SXM   partition=gpu,     time=360, gres=gpu:a100sxm:1,  nodes: sacramento
  A100_PCIe  partition=gpu_top, time=360, gres=gpu:a100pcie:1, nodes: fresko
  H100_SXM   partition=gpu_top, time=360, gres=gpu:h100pcie:1, nodes: sanjose (auto-retry on GPU mismatch)
  H100_PCIe  partition=gpu_top, time=360, gres=gpu:h100pcie:1, nodes: sanjose (auto-retry on GPU mismatch)
  H200       partition=gpu_top, time=360, gres=gpu:h200sxm:1,  nodes: trinity
  L40S       partition=gpu_top, time=360, gres=gpu:l40spcie:1, nodes: fresko

Slurm scheduling:
  Jobs are submitted with the full node list. Slurm's scheduler picks
  whichever node has resources available first, avoiding idle nodes
  while others are overloaded.

Examples:
  # 50 models across 3 V100 nodes (Slurm picks the free one):
  ./run_benchmark_gpu.sh --gpu V100 --config config-v2.yaml

  # Single H200 node:
  ./run_benchmark_gpu.sh --gpu H200 --config config-v2.yaml

  # Only 2 of 3 V100 nodes:
  ./run_benchmark_gpu.sh --gpu V100 --config config-v2.yaml --nodes losangeles,sanfrancisco

  # Dry run:
  ./run_benchmark_gpu.sh --gpu V100 --config config-v2.yaml --dry-run

  # Resume after failure:
  ./run_benchmark_gpu.sh --gpu V100 --resume temp_configs_gpu_config-v2_V100
EOF
}

# --- Argument Parsing ---
GPU_TYPE=""
CONFIG=""
RESUME_DIR=""
ACCOUNT="init"
BATCH_SIZE=1
MAX_JOBS=500
NODES_OVERRIDE=""
MAX_RETRIES=10
EXCLUSIVE=false
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)         GPU_TYPE="$2";       shift 2 ;;
        --config)      CONFIG="$2";         shift 2 ;;
        --resume)      RESUME_DIR="$2";     shift 2 ;;
        --account)     ACCOUNT="$2";        shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";     shift 2 ;;
        --max-jobs)    MAX_JOBS="$2";       shift 2 ;;
        --nodes)       NODES_OVERRIDE="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2";    shift 2 ;;
        --exclusive)   EXCLUSIVE=true;      shift ;;
        --dry-run)     DRY_RUN=true;        shift ;;
        -h|--help)     usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# --- Validation ---
if [ -z "$GPU_TYPE" ]; then
    echo "Error: --gpu is required." >&2
    usage >&2
    exit 1
fi

if [ -n "$CONFIG" ] && [ -n "$RESUME_DIR" ]; then
    echo "Error: --config and --resume are mutually exclusive." >&2
    exit 1
fi

if [ -z "$CONFIG" ] && [ -z "$RESUME_DIR" ]; then
    echo "Error: Either --config or --resume is required." >&2
    usage >&2
    exit 1
fi

if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' does not exist." >&2
    exit 1
fi

if [ -n "$RESUME_DIR" ] && [ ! -d "$RESUME_DIR" ]; then
    echo "Error: Resume directory '$RESUME_DIR' does not exist." >&2
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -eq 0 ]; then
    echo "Error: --batch-size must be a positive integer." >&2
    exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_JOBS" -eq 0 ]; then
    echo "Error: --max-jobs must be a positive integer." >&2
    exit 1
fi

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "Error: --max-retries must be a non-negative integer." >&2
    exit 1
fi

# --- Resolve GPU Mapping ---
GPU_PARTITION=""
GPU_TIME=""
GPU_GRES=""
GPU_NODES=""

case "$GPU_TYPE" in
    V100)
        GPU_PARTITION="gpu"
        GPU_TIME=360
        GPU_GRES="gpu:v100sxm:1"
        GPU_NODES="losangeles,sanfrancisco,sandiego"
        ;;
    A100_SXM)
        GPU_PARTITION="gpu"
        GPU_TIME=360
        GPU_GRES="gpu:a100sxm:1"
        GPU_NODES="sacramento"
        ;;
    A100_PCIe)
        GPU_PARTITION="gpu_top"
        GPU_TIME=360
        GPU_GRES="gpu:a100pcie:1"
        GPU_NODES="fresko"
        ;;
    H100_SXM)
        GPU_PARTITION="gpu_top"
        GPU_TIME=360
        GPU_GRES="gpu:h100pcie:1"
        GPU_NODES="sanjose"
        ;;
    H100_PCIe)
        GPU_PARTITION="gpu_top"
        GPU_TIME=360
        GPU_GRES="gpu:h100pcie:1"
        GPU_NODES="sanjose"
        ;;
    H200)
        GPU_PARTITION="gpu_top"
        GPU_TIME=360
        GPU_GRES="gpu:h200sxm:1"
        GPU_NODES="trinity"
        ;;
    L40S)
        GPU_PARTITION="gpu_top"
        GPU_TIME=360
        GPU_GRES="gpu:l40spcie:1"
        GPU_NODES="fresko"
        ;;
    *)
        echo "Error: Unknown GPU type '$GPU_TYPE'. Must be one of: V100, A100_SXM, A100_PCIe, H100_SXM, H100_PCIe, H200, L40S" >&2
        exit 1
        ;;
esac

# --- Determine expected GPU for validation ---
EXPECTED_GPU=""
case "$GPU_TYPE" in
    H100_SXM|H100_PCIe)
        EXPECTED_GPU="$GPU_TYPE"
        ;;
esac

# --- Apply --nodes override ---
if [ -n "$NODES_OVERRIDE" ]; then
    # Validate each overridden node is in the GPU's node list
    IFS=',' read -ra OVERRIDE_ARRAY <<< "$NODES_OVERRIDE"
    for node in "${OVERRIDE_ARRAY[@]}"; do
        if ! echo ",$GPU_NODES," | grep -q ",$node,"; then
            echo "Error: Node '$node' is not valid for GPU type '$GPU_TYPE'. Valid nodes: $GPU_NODES" >&2
            exit 1
        fi
    done
    GPU_NODES="$NODES_OVERRIDE"
fi

# --- Auto-derive paths ---
if [ -n "$CONFIG" ]; then
    CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
else
    CONFIG_BASENAME=$(basename "$RESUME_DIR")
fi

TEMP_DIR="temp_configs_gpu_${CONFIG_BASENAME}_${GPU_TYPE}"
LOG_FILE="run_benchmark_gpu_${CONFIG_BASENAME}_${GPU_TYPE}.log"
ABSOLUTE_LOG_FILE="$(pwd)/$LOG_FILE"

# Redirect all stdout/stderr (including child processes) to both console and log file
exec > >(tee -a "$ABSOLUTE_LOG_FILE") 2>&1

# --- Logging ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Function to Create Batch Configs ---
create_batch_configs() {
    local main_config_path=$1
    local batch_size=$2
    local output_dir=$3

    if [ ! -f "$main_config_path" ]; then
        log_message "Error: Main config file not found at $main_config_path"
        exit 1
    fi

    mkdir -p "$output_dir"

    mapfile -t all_models < <(sed -n '/^LLM_MODELS:/,/^[^[:space:]]/p' "$main_config_path" | grep '  - ' | sed 's/  - //;s/\r$//')

    if [ ${#all_models[@]} -eq 0 ]; then
        log_message "Warning: No 'LLM_MODELS' found in the config file."
        return
    fi

    config_prefix=$(awk '/^LLM_MODELS:/ {exit} {print}' "$main_config_path")
    config_suffix=$(awk 'f{print} /^[^[:space:]]/ && p {f=1} /^LLM_MODELS:/ {p=1}' "$main_config_path")

    local num_models=${#all_models[@]}
    local num_batches=$(( (num_models + batch_size - 1) / batch_size ))

    log_message "Splitting $num_models models into $num_batches batches of (up to) $batch_size models each."

    for (( i=0; i<num_batches; i++ )); do
        local batch_start=$((i * batch_size))
        local batch_models=("${all_models[@]:batch_start:batch_size}")
        local batch_config_filename
        batch_config_filename=$(printf "%s/config_batch_%03d.yaml" "$output_dir" $((i+1)))

        {
            echo "$config_prefix"
            echo "LLM_MODELS:"
            for model in "${batch_models[@]}"; do
                echo "  - $model"
            done
            if [ -n "$config_suffix" ]; then
                echo "$config_suffix"
            fi
        } > "$batch_config_filename"
    done
}

# --- 0. Log startup + effective configuration ---
log_message "=== GPU Benchmark Wrapper Started ==="
log_message "Configuration:"
log_message "  GPU type:     $GPU_TYPE"
log_message "  Partition:    $GPU_PARTITION"
log_message "  Time limit:   $GPU_TIME minutes"
[ -n "$GPU_GRES" ] && log_message "  GRES:         $GPU_GRES"
log_message "  Account:      $ACCOUNT"
log_message "  Nodes:        $GPU_NODES"
log_message "  Batch size:   $BATCH_SIZE"
log_message "  Max jobs:     $MAX_JOBS"
log_message "  Temp dir:     $TEMP_DIR"
log_message "  Log file:     $LOG_FILE"
[ -n "$EXPECTED_GPU" ] && log_message "  Expected GPU: $EXPECTED_GPU"
[ -n "$EXPECTED_GPU" ] && log_message "  Max retries:  $MAX_RETRIES"
log_message "  Exclusive:    $EXCLUSIVE"
log_message "  Dry run:      $DRY_RUN"

# --- 1. Generate configs or use existing resume dir ---
if [ -n "$RESUME_DIR" ]; then
    TEMP_DIR="$RESUME_DIR"
    log_message "Resuming from provided temp directory: $TEMP_DIR"
    if [ ! -d "$TEMP_DIR" ]; then
        log_message "Error: Resume directory '$TEMP_DIR' does not exist."
        exit 1
    fi

    remaining=$(find "$TEMP_DIR" -maxdepth 1 -name "config_batch_*.yaml" -type f | wc -l)
    if [ "$remaining" -eq 0 ]; then
        log_message "No remaining configs found in $TEMP_DIR. Nothing to do."
        exit 0
    fi
    log_message "Found $remaining config(s) remaining."
else
    log_message "Starting a new run. Generating batch configs from $CONFIG..."
    create_batch_configs "$CONFIG" "$BATCH_SIZE" "$TEMP_DIR"

    remaining=$(find "$TEMP_DIR" -maxdepth 1 -name "config_batch_*.yaml" -type f | wc -l)
    if [ "$remaining" -eq 0 ]; then
        log_message "Batch config generation failed or produced no files. Exiting."
        exit 1
    fi
    log_message "Generated $remaining batch config(s)."
fi

# --- 2. Launch single run_all_slurm.sh with full node list ---
SLURM_LOG="run_all_slurm_${CONFIG_BASENAME}_${GPU_TYPE}.log"

CMD="./run_all_slurm.sh"
CMD="$CMD --resume $TEMP_DIR"
CMD="$CMD --partition $GPU_PARTITION"
CMD="$CMD --nodelist $GPU_NODES"
CMD="$CMD --time $GPU_TIME"
CMD="$CMD --account $ACCOUNT"
CMD="$CMD --max-jobs $MAX_JOBS"
CMD="$CMD --log $SLURM_LOG"
[ -n "$GPU_GRES" ] && CMD="$CMD --gres $GPU_GRES"
[ -n "$EXPECTED_GPU" ] && CMD="$CMD --expected-gpu $EXPECTED_GPU --max-retries $MAX_RETRIES"
[ "$EXCLUSIVE" = true ] && CMD="$CMD --exclusive"
[ "$DRY_RUN" = true ] && CMD="$CMD --dry-run"

log_message "Launching: $CMD"
$CMD
EXIT_CODE=$?

# --- 3. Cleanup ---
if [ "$EXIT_CODE" -eq 0 ] && [ -d "$TEMP_DIR" ] && [ -z "$(ls -A "$TEMP_DIR" 2>/dev/null)" ]; then
    log_message "All configs processed. Cleaning up $TEMP_DIR..."
    rm -rf "$TEMP_DIR"
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    log_message "=== All jobs completed successfully ==="
else
    log_message "=== run_all_slurm.sh FAILED with exit code $EXIT_CODE ==="
    exit "$EXIT_CODE"
fi
