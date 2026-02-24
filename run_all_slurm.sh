#!/bin/bash

# Parameterized SLURM orchestration script.
# Replaces run_all_slurm_batch.sh, run_all_slurm_batch_l40.sh, and
# run_all_slurm_batch_qwen_H200.sh with a single invocation per node.

set -euo pipefail

# --- Usage ---
usage() {
    cat <<'EOF'
Usage:
  ./run_all_slurm.sh --config <file> --partition <name> --nodelist <nodes> --time <minutes> [options]
  ./run_all_slurm.sh --resume <dir>  --partition <name> --nodelist <nodes> --time <minutes> [options]

Required flags:
  --config <file>       Main YAML config file (mutually exclusive with --resume)
  --partition <name>    SLURM partition (e.g., gpu, gpu_top)
  --nodelist <nodes>    Comma-separated list of target nodes (e.g., losangeles,sanfrancisco)
  --time <minutes>      Job time limit in minutes

Optional flags:
  --account <name>      SLURM account (e.g., init)
  --gres <spec>         GPU resource spec (e.g., gpu:l40spcie:1)
  --resume <dir>        Resume from existing temp config dir (mutually exclusive with --config)
  --batch-size <n>      Models per batch config (default: 1)
  --max-jobs <n>        Max concurrent SLURM jobs before throttling (default: 500)
  --log <file>          Log file path (default: auto-derived from config basename)
  --temp-dir <dir>      Temp config directory (default: auto-derived from config basename)
  --expected-gpu <type>  Expected GPU variant (e.g., H100_SXM, H100_PCIe) for validation
  --max-retries <n>     Max retries on GPU mismatch, exit code 42 (default: 0)
  --exclusive           Request exclusive node access (jobs run sequentially, safer for shared nodes)
  --dry-run             Print sbatch commands without executing
  -h, --help            Show this help message

Examples:
  # Run on a single GPU node:
  ./run_all_slurm.sh --config config-50-3.yaml --partition gpu --nodelist sanfrancisco --time 4320

  # With account and specific GPU type:
  ./run_all_slurm.sh --config config-Qwen2.5-0.5B-Instruct.yaml --partition gpu_top --nodelist trinity --time 1440 --account init

  # With L40S GPU resource spec:
  ./run_all_slurm.sh --config config-50-3.yaml --partition gpu_top --nodelist fresko --time 1440 --gres gpu:l40spcie:1

  # Dry run (show sbatch commands without submitting):
  ./run_all_slurm.sh --config config-50-3.yaml --partition gpu --nodelist sanfrancisco --time 4320 --dry-run

  # Resume a failed run:
  ./run_all_slurm.sh --resume temp_configs_config-50-3 --partition gpu --nodelist sanfrancisco --time 4320
EOF
}

# --- Argument Parsing ---
CONFIG=""
PARTITION=""
NODELIST=""
TIME=""
ACCOUNT=""
GRES=""
RESUME_DIR=""
BATCH_SIZE=1
MAX_JOBS=500
LOG_FILE=""
TEMP_CONFIG_DIR=""
EXPECTED_GPU=""
MAX_RETRIES=0
EXCLUSIVE=false
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --partition)   PARTITION="$2";    shift 2 ;;
        --nodelist)    NODELIST="$2";     shift 2 ;;
        --time)        TIME="$2";        shift 2 ;;
        --account)     ACCOUNT="$2";     shift 2 ;;
        --gres)        GRES="$2";        shift 2 ;;
        --resume)      RESUME_DIR="$2";  shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --max-jobs)    MAX_JOBS="$2";    shift 2 ;;
        --log)         LOG_FILE="$2";    shift 2 ;;
        --temp-dir)    TEMP_CONFIG_DIR="$2"; shift 2 ;;
        --expected-gpu) EXPECTED_GPU="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --exclusive)   EXCLUSIVE=true;   shift ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        -h|--help)     usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# --- Validation ---
if [ -n "$CONFIG" ] && [ -n "$RESUME_DIR" ]; then
    echo "Error: --config and --resume are mutually exclusive." >&2
    exit 1
fi

if [ -z "$CONFIG" ] && [ -z "$RESUME_DIR" ]; then
    echo "Error: Either --config or --resume is required." >&2
    usage >&2
    exit 1
fi

if [ -z "$PARTITION" ]; then
    echo "Error: --partition is required." >&2
    exit 1
fi

if [ -z "$NODELIST" ]; then
    echo "Error: --nodelist is required." >&2
    exit 1
fi

if [ -z "$TIME" ]; then
    echo "Error: --time is required." >&2
    exit 1
fi

# Numeric checks
if ! [[ "$TIME" =~ ^[0-9]+$ ]]; then
    echo "Error: --time must be a positive integer (minutes)." >&2
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

if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' does not exist." >&2
    exit 1
fi

if [ -n "$RESUME_DIR" ] && [ ! -d "$RESUME_DIR" ]; then
    echo "Error: Resume directory '$RESUME_DIR' does not exist." >&2
    exit 1
fi

# --- Auto-derive defaults ---
if [ -n "$CONFIG" ]; then
    CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
else
    CONFIG_BASENAME=$(basename "$RESUME_DIR")
fi

if [ -z "$TEMP_CONFIG_DIR" ]; then
    TEMP_CONFIG_DIR="temp_configs_${CONFIG_BASENAME}"
fi

if [ -z "$LOG_FILE" ]; then
    LOG_FILE="run_all_slurm_${CONFIG_BASENAME}.log"
fi

# Build sbatch command
SBATCH_CMD="sbatch --partition=$PARTITION --nodelist=$NODELIST --time=$TIME"
[ -n "$ACCOUNT" ]       && SBATCH_CMD="$SBATCH_CMD --account=$ACCOUNT"
[ -n "$GRES" ]          && SBATCH_CMD="$SBATCH_CMD --gres=$GRES"
[ "$EXCLUSIVE" = true ] && SBATCH_CMD="$SBATCH_CMD --exclusive=user"

# Build cleanup sbatch base (lightweight job, use cpu partition if available)
CLEANUP_SBATCH_CMD="sbatch --partition=cpu"
[ -n "$ACCOUNT" ] && CLEANUP_SBATCH_CMD="$CLEANUP_SBATCH_CMD --account=$ACCOUNT"

# Get the absolute path for the log file
ABSOLUTE_LOG_FILE="$(pwd)/$LOG_FILE"

# Redirect all stdout/stderr (including child processes) to both console and log file
exec > >(tee -a "$ABSOLUTE_LOG_FILE") 2>&1

# --- Function to log messages ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Function to Create Batch Configs ---
create_batch_configs() {
    local main_config_path=$1
    local batch_size=$2

    if [ ! -f "$main_config_path" ]; then
        log_message "Error: Main config file not found at $main_config_path"
        exit 1
    fi

    log_message "Creating temporary config directory: $TEMP_CONFIG_DIR"
    mkdir -p "$TEMP_CONFIG_DIR"

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
        batch_config_filename=$(printf "%s/config_batch_%03d.yaml" "$TEMP_CONFIG_DIR" $((i+1)))

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

        echo "  -> Created $batch_config_filename"
    done
}

# --- 0. Log startup + effective configuration ---
log_message "--- Orchestration Script Started ---"
log_message "Configuration:"
log_message "  Partition:    $PARTITION"
log_message "  Nodes:        $NODELIST"
log_message "  Time limit:   $TIME minutes"
[ -n "$ACCOUNT" ] && log_message "  Account:      $ACCOUNT"
[ -n "$GRES" ]    && log_message "  GRES:         $GRES"
log_message "  Batch size:   $BATCH_SIZE"
log_message "  Max jobs:     $MAX_JOBS"
log_message "  Temp dir:     $TEMP_CONFIG_DIR"
log_message "  Log file:     $LOG_FILE"
[ -n "$EXPECTED_GPU" ] && log_message "  Expected GPU: $EXPECTED_GPU"
[ -n "$EXPECTED_GPU" ] && log_message "  Max retries:  $MAX_RETRIES"
log_message "  Exclusive:    $EXCLUSIVE"
log_message "  Dry run:      $DRY_RUN"
log_message "  sbatch cmd:   $SBATCH_CMD"

# --- Write retry state file if GPU validation is enabled ---
RETRY_STATE_FILE=""
if [ -n "$EXPECTED_GPU" ] && [ "$MAX_RETRIES" -gt 0 ]; then
    RETRY_STATE_FILE="$(pwd)/.retry_state_${CONFIG_BASENAME}.env"
    cat > "$RETRY_STATE_FILE" <<STATEEOF
SBATCH_CMD="$SBATCH_CMD"
CLEANUP_SBATCH_CMD="$CLEANUP_SBATCH_CMD"
EXPECTED_GPU="$EXPECTED_GPU"
CONFIG_BASENAME="$CONFIG_BASENAME"
MAX_RETRIES="$MAX_RETRIES"
LOG_FILE="$ABSOLUTE_LOG_FILE"
STATEEOF
    log_message "Wrote retry state file: $RETRY_STATE_FILE"
fi

# --- 1. Generate or resume batch configs ---
if [ -n "$RESUME_DIR" ]; then
    TEMP_CONFIG_DIR="$RESUME_DIR"
    log_message "Resuming from provided temp directory: $TEMP_CONFIG_DIR"
    if [ ! -d "$TEMP_CONFIG_DIR" ]; then
        log_message "Error: Provided temp directory '$TEMP_CONFIG_DIR' does not exist."
        exit 1
    fi
else
    log_message "Starting a new run. Generating batch configs from $CONFIG..."
    create_batch_configs "$CONFIG" "$BATCH_SIZE"

    if [ ! -d "$TEMP_CONFIG_DIR" ] || [ -z "$(ls -A "$TEMP_CONFIG_DIR")" ]; then
        log_message "Batch config generation failed or produced no files. Exiting."
        exit 1
    fi
fi

# --- 2. Submit one SLURM job per batch config ---
log_message "Preparing per-job SLURM submission..."

batch_configs=()
while IFS= read -r f; do batch_configs+=("$f"); done \
  < <(find "$TEMP_CONFIG_DIR" -name "config_batch_*.yaml" -type f | sort)

if [ ${#batch_configs[@]} -eq 0 ]; then
    log_message "No batch configs found in $TEMP_CONFIG_DIR. Nothing to submit."
    exit 0
fi

log_message "Found ${#batch_configs[@]} batch config(s) to submit (max $MAX_JOBS concurrent)."

EXPORT_VARS="ALL,CONFIG_BASENAME=$CONFIG_BASENAME"
[ -n "$EXPECTED_GPU" ] && EXPORT_VARS="${EXPORT_VARS},EXPECTED_GPU=$EXPECTED_GPU"
JOB_SUBMIT_CMD="$SBATCH_CMD --export=$EXPORT_VARS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for batch_config in "${batch_configs[@]}"; do
    # Extract model slug from config for the job name (visible in squeue)
    model_slug=$(awk '/^LLM_MODELS:/ { in_b=1; next }
                      in_b && /^\s*-\s*"/ {
                          gsub(/^[^"]*"/, ""); gsub(/".*$/, "")
                          n = split($0, a, "/"); print a[n]; exit
                      }' "$batch_config" \
                 | sed 's/[^A-Za-z0-9._-]/_/g' | cut -c1-40)
    [ -z "$model_slug" ] && model_slug="$(basename "$batch_config" .yaml)"

    if [ "$DRY_RUN" = true ]; then
        log_message "[DRY RUN] $JOB_SUBMIT_CMD --job-name=${model_slug} ./run_benchmark_slurm.submit $batch_config"
        continue
    fi

    # Throttle: wait until we are under the max-jobs limit
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$MAX_JOBS" ]; do
        log_message "At max jobs ($MAX_JOBS). Waiting 60s..."
        sleep 60
    done

    job_output=$($JOB_SUBMIT_CMD --job-name="${model_slug}" ./run_benchmark_slurm.submit "$batch_config")
    job_id=$(echo "$job_output" | awk '{print $4}')
    log_message "SUBMITTED job $job_id for $(basename "$batch_config")"

    # Small delay between submissions to avoid hammering the SLURM controller
    sleep $(( 5 + RANDOM % 6 ))

    dep_spec="afterany:$job_id"
    if [ -n "$RETRY_STATE_FILE" ]; then
        retry_cmd="$SCRIPT_DIR/retry_on_gpu_mismatch.sh $job_id $batch_config 1 $RETRY_STATE_FILE"
        $CLEANUP_SBATCH_CMD --dependency="$dep_spec" \
          --job-name="retry_$(basename "$batch_config" .yaml)" \
          --output=/dev/null --error=/dev/null \
          --wrap="$retry_cmd"
    else
        cleanup_command="echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] COMPLETED: $batch_config\" >> \"$ABSOLUTE_LOG_FILE\"; rm '$batch_config'"
        $CLEANUP_SBATCH_CMD --dependency="$dep_spec" \
          --job-name="cleanup_$(basename "$batch_config" .yaml)" \
          --output=/dev/null --error=/dev/null \
          --wrap="$cleanup_command"
    fi
done

log_message "All batch jobs have been submitted and are being managed by the queue."

if [ "$DRY_RUN" = true ]; then
    log_message "--- Dry Run Complete (no jobs were submitted) ---"
    exit 0
fi

# --- 3. Final Check ---
while true; do
    current_jobs=$(squeue -u "$USER" -h | wc -l)
    if [ "$current_jobs" -eq 0 ]; then
        log_message "All jobs have completed."
        break
    else
        remaining_configs=$(find "$TEMP_CONFIG_DIR" -name "config_batch_*.yaml" -type f | wc -l)
        log_message "Waiting for the final $current_jobs job(s) to complete... ($remaining_configs config(s) remaining)"
        sleep 60
    fi
done

# --- 4. Cleanup ---
if [ -d "$TEMP_CONFIG_DIR" ] && [ -z "$(ls -A "$TEMP_CONFIG_DIR")" ]; then
    log_message "Cleaning up empty temporary config directory..."
    rm -rf "$TEMP_CONFIG_DIR"
fi

log_message "--- Orchestration Complete ---"
