#!/bin/bash

# GPU-aware wrapper for SLURM orchestration.
# Resolves SLURM parameters from a GPU mapping table and distributes
# models round-robin across multiple nodes of the same GPU type.

set -euo pipefail

# --- Usage ---
usage() {
    cat <<'EOF'
Usage:
  ./run_benchmark_gpu.sh --gpu <type> --config <file> [options]
  ./run_benchmark_gpu.sh --gpu <type> --resume <dir>  [options]

Required flags:
  --gpu <type>          GPU type: V100, A100_SXM, A100_PCIe, H100, H200, L40S
  --config <file>       Main YAML config file (mutually exclusive with --resume)

Optional flags:
  --resume <dir>        Resume from existing temp dir (mutually exclusive with --config)
  --account <name>      SLURM account (default: init)
  --batch-size <n>      Models per batch config (default: 1)
  --max-jobs <n>        Max concurrent SLURM jobs per node (default: 500)
  --nodes <list>        Comma-separated node subset override
  --dry-run             Pass through to run_all_slurm.sh
  -h, --help            Show this help message

GPU types and their defaults:
  V100       partition=gpu,     time=4320, nodes: losangeles,sanfrancisco,sandiego
  A100_SXM   partition=gpu,     time=4320, nodes: sacramento
  A100_PCIe  partition=gpu_top, time=1440, gres=gpu:a100pcie:1, nodes: fresko
  H100       partition=gpu_top, time=1440, nodes: sanjose
  H200       partition=gpu_top, time=1440, nodes: trinity
  L40S       partition=gpu_top, time=1440, gres=gpu:l40spcie:1, nodes: fresko

Examples:
  # 50 models across 3 V100 nodes (~17 per node):
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

# --- Resolve GPU Mapping ---
GPU_PARTITION=""
GPU_TIME=""
GPU_GRES=""
GPU_NODES=""

case "$GPU_TYPE" in
    V100)
        GPU_PARTITION="gpu"
        GPU_TIME=4320
        GPU_GRES=""
        GPU_NODES="losangeles,sanfrancisco,sandiego"
        ;;
    A100_SXM)
        GPU_PARTITION="gpu"
        GPU_TIME=4320
        GPU_GRES=""
        GPU_NODES="sacramento"
        ;;
    A100_PCIe)
        GPU_PARTITION="gpu_top"
        GPU_TIME=1440
        GPU_GRES="gpu:a100pcie:1"
        GPU_NODES="fresko"
        ;;
    H100)
        GPU_PARTITION="gpu_top"
        GPU_TIME=1440
        GPU_GRES=""
        GPU_NODES="sanjose"
        ;;
    H200)
        GPU_PARTITION="gpu_top"
        GPU_TIME=1440
        GPU_GRES=""
        GPU_NODES="trinity"
        ;;
    L40S)
        GPU_PARTITION="gpu_top"
        GPU_TIME=1440
        GPU_GRES="gpu:l40spcie:1"
        GPU_NODES="fresko"
        ;;
    *)
        echo "Error: Unknown GPU type '$GPU_TYPE'. Must be one of: V100, A100_SXM, A100_PCIe, H100, H200, L40S" >&2
        exit 1
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

# Build node array
IFS=',' read -ra NODE_ARRAY <<< "$GPU_NODES"

# --- Auto-derive paths ---
if [ -n "$CONFIG" ]; then
    CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
else
    CONFIG_BASENAME=$(basename "$RESUME_DIR")
fi

TOP_TEMP_DIR="temp_configs_gpu_${CONFIG_BASENAME}_${GPU_TYPE}"
LOG_FILE="run_benchmark_gpu_${CONFIG_BASENAME}_${GPU_TYPE}.log"
ABSOLUTE_LOG_FILE="$(pwd)/$LOG_FILE"

# Redirect all stdout/stderr (including child processes) to both console and log file
exec > >(tee -a "$ABSOLUTE_LOG_FILE") 2>&1

# --- Logging ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Function to Create Batch Configs (into a staging dir) ---
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
log_message "  Nodes:        ${NODE_ARRAY[*]}"
log_message "  Batch size:   $BATCH_SIZE"
log_message "  Max jobs:     $MAX_JOBS"
log_message "  Temp dir:     $TOP_TEMP_DIR"
log_message "  Log file:     $LOG_FILE"
log_message "  Dry run:      $DRY_RUN"

# --- 1. Generate & distribute configs or discover remaining work ---
if [ -n "$RESUME_DIR" ]; then
    TOP_TEMP_DIR="$RESUME_DIR"
    log_message "Resuming from provided temp directory: $TOP_TEMP_DIR"
    if [ ! -d "$TOP_TEMP_DIR" ]; then
        log_message "Error: Resume directory '$TOP_TEMP_DIR' does not exist."
        exit 1
    fi

    # Discover which nodes still have work
    ACTIVE_NODES=()
    for node in "${NODE_ARRAY[@]}"; do
        node_dir="$TOP_TEMP_DIR/$node"
        if [ -d "$node_dir" ]; then
            remaining=$(find "$node_dir" -name "config_batch_*.yaml" -type f | wc -l)
            if [ "$remaining" -gt 0 ]; then
                ACTIVE_NODES+=("$node")
                log_message "  Node $node: $remaining config(s) remaining"
            else
                log_message "  Node $node: completed (no configs remaining)"
            fi
        else
            log_message "  Node $node: no directory found, skipping"
        fi
    done

    if [ ${#ACTIVE_NODES[@]} -eq 0 ]; then
        log_message "No remaining work found in any node subdirectory. Nothing to do."
        exit 0
    fi
else
    log_message "Starting a new run. Generating batch configs from $CONFIG..."

    # Generate into a staging directory
    STAGING_DIR="${TOP_TEMP_DIR}/_staging"
    create_batch_configs "$CONFIG" "$BATCH_SIZE" "$STAGING_DIR"

    # Count generated configs
    mapfile -t BATCH_FILES < <(find "$STAGING_DIR" -name "config_batch_*.yaml" -type f | sort)

    if [ ${#BATCH_FILES[@]} -eq 0 ]; then
        log_message "Batch config generation failed or produced no files. Exiting."
        rm -rf "$STAGING_DIR"
        exit 1
    fi

    # Create per-node subdirectories
    for node in "${NODE_ARRAY[@]}"; do
        mkdir -p "$TOP_TEMP_DIR/$node"
    done

    # Distribute configs round-robin across nodes
    num_nodes=${#NODE_ARRAY[@]}
    for (( i=0; i<${#BATCH_FILES[@]}; i++ )); do
        node_idx=$(( i % num_nodes ))
        target_node="${NODE_ARRAY[$node_idx]}"
        mv "${BATCH_FILES[$i]}" "$TOP_TEMP_DIR/$target_node/"
    done

    # Remove staging directory
    rm -rf "$STAGING_DIR"

    # Log distribution
    log_message "Distributed ${#BATCH_FILES[@]} batch configs across ${num_nodes} node(s):"
    ACTIVE_NODES=()
    for node in "${NODE_ARRAY[@]}"; do
        count=$(find "$TOP_TEMP_DIR/$node" -name "config_batch_*.yaml" -type f | wc -l)
        log_message "  $node: $count config(s)"
        if [ "$count" -gt 0 ]; then
            ACTIVE_NODES+=("$node")
        fi
    done
fi

# --- 2. Launch run_all_slurm.sh per node in background ---
log_message "Launching run_all_slurm.sh for ${#ACTIVE_NODES[@]} node(s)..."

declare -A NODE_PIDS

for node in "${ACTIVE_NODES[@]}"; do
    node_dir="$TOP_TEMP_DIR/$node"
    node_log="run_all_slurm_${CONFIG_BASENAME}_${GPU_TYPE}_${node}.log"

    # Build the command
    CMD="./run_all_slurm.sh"
    CMD="$CMD --resume $node_dir"
    CMD="$CMD --partition $GPU_PARTITION"
    CMD="$CMD --nodelist $node"
    CMD="$CMD --time $GPU_TIME"
    CMD="$CMD --account $ACCOUNT"
    CMD="$CMD --max-jobs $MAX_JOBS"
    CMD="$CMD --log $node_log"
    [ -n "$GPU_GRES" ] && CMD="$CMD --gres $GPU_GRES"
    [ "$DRY_RUN" = true ] && CMD="$CMD --dry-run"

    log_message "  [$node] $CMD"
    $CMD &
    NODE_PIDS[$node]=$!
done

# --- 3. Wait for all background PIDs ---
log_message "Waiting for all node orchestrators to complete..."

ALL_OK=true
for node in "${ACTIVE_NODES[@]}"; do
    pid=${NODE_PIDS[$node]}
    if wait "$pid"; then
        log_message "  [$node] completed successfully (PID $pid)"
    else
        exit_code=$?
        log_message "  [$node] FAILED with exit code $exit_code (PID $pid)"
        ALL_OK=false
    fi
done

# --- 4. Cleanup ---
if [ "$ALL_OK" = true ] && [ -d "$TOP_TEMP_DIR" ]; then
    # Check if all node subdirs are empty
    all_empty=true
    for node in "${NODE_ARRAY[@]}"; do
        node_dir="$TOP_TEMP_DIR/$node"
        if [ -d "$node_dir" ] && [ -n "$(ls -A "$node_dir" 2>/dev/null)" ]; then
            all_empty=false
            break
        fi
    done

    if [ "$all_empty" = true ]; then
        log_message "All node subdirectories are empty. Cleaning up $TOP_TEMP_DIR..."
        rm -rf "$TOP_TEMP_DIR"
    else
        log_message "Some node subdirectories still contain configs (incomplete runs). Keeping $TOP_TEMP_DIR for resume."
    fi
fi

if [ "$ALL_OK" = true ]; then
    log_message "=== All nodes completed successfully ==="
else
    log_message "=== Some nodes FAILED — check per-node logs ==="
    exit 1
fi
