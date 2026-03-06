#!/bin/bash

# Retry wrapper for GPU mismatch (exit code 42).
# Called as a SLURM cleanup job after the main benchmark job completes.
#
# Usage:
#   retry_on_gpu_mismatch.sh <job_id> <batch_config> <retry_count> <state_file>
#
# Arguments:
#   job_id         - SLURM job ID of the completed benchmark job
#   batch_config   - Path to the batch config YAML file
#   retry_count    - Current retry attempt number (starts at 1)
#   state_file     - Path to shared state file containing SBATCH_CMD, CLEANUP_SBATCH_CMD,
#                    EXPECTED_GPU, MAX_RETRIES, LOG_FILE

set -uo pipefail

JOB_ID="$1"
BATCH_CONFIG="$2"
RETRY_COUNT="$3"
STATE_FILE="$4"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load shared state (SBATCH_CMD, CLEANUP_SBATCH_CMD, EXPECTED_GPU, MAX_RETRIES, LOG_FILE)
source "$STATE_FILE"

# Derive model slug for job naming (mirrors logic in run_all_slurm.sh)
MODEL_SLUG=$(awk '/^LLM_MODELS:/ { in_b=1; next }
                  in_b && /^\s*-\s*"/ {
                      gsub(/^[^"]*"/, ""); gsub(/".*$/, "")
                      n = split($0, a, "/"); print a[n]; exit
                  }' "$BATCH_CONFIG" \
             | sed 's/[^A-Za-z0-9._-]/_/g' | cut -c1-40)
[ -z "$MODEL_SLUG" ] && MODEL_SLUG="$(basename "$BATCH_CONFIG" .yaml)"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Get exit code of the completed job via sacct
# Use the .batch step which reflects the actual script exit code
EXIT_CODE=$(sacct -j "$JOB_ID.batch" --format=ExitCode --noheader --parsable2 | head -n 1 | cut -d: -f1)

log_message "Retry wrapper: Job $JOB_ID exited with code $EXIT_CODE (retry $RETRY_COUNT/$MAX_RETRIES)"

if [ "$EXIT_CODE" -eq 0 ]; then
    log_message "COMPLETED: $BATCH_CONFIG (job $JOB_ID succeeded)"
    rm -f "$BATCH_CONFIG"
    exit 0
fi

if [ "$EXIT_CODE" -eq 42 ]; then
    if [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
        NEXT_RETRY=$((RETRY_COUNT + 1))
        log_message "GPU mismatch on attempt $RETRY_COUNT/$MAX_RETRIES. Resubmitting $BATCH_CONFIG (attempt $NEXT_RETRY)..."

        # Wait a random delay (30-60 min) before retrying to spread out retries
        # and give the scheduler time to potentially assign a different GPU
        RETRY_DELAY=$(( 1800 + RANDOM % 1801 ))
        log_message "Waiting ${RETRY_DELAY}s before resubmitting..."
        sleep "$RETRY_DELAY"

        # Resubmit the benchmark job with EXPECTED_GPU env var
        job_output=$($SBATCH_CMD --job-name="${MODEL_SLUG}" --export=ALL,EXPECTED_GPU="$EXPECTED_GPU",CONFIG_BASENAME="$CONFIG_BASENAME" "$SCRIPT_DIR/run_benchmark_slurm.submit" "$BATCH_CONFIG")
        if [ $? -ne 0 ]; then
            log_message "FAILED to resubmit $BATCH_CONFIG on retry $NEXT_RETRY"
            exit 1
        fi

        new_job_id=$(echo "$job_output" | awk '{print $4}')
        log_message "Resubmitted $BATCH_CONFIG as job $new_job_id (attempt $NEXT_RETRY/$MAX_RETRIES)"

        # Submit a new cleanup/retry job that depends on the resubmitted job
        $CLEANUP_SBATCH_CMD \
            --dependency=afterany:"$new_job_id" \
            --job-name="retry_$(basename "$BATCH_CONFIG" .yaml)" \
            --output=/dev/null --error=/dev/null \
            --wrap="$SCRIPT_DIR/retry_on_gpu_mismatch.sh $new_job_id $BATCH_CONFIG $NEXT_RETRY $STATE_FILE"

        exit 0
    else
        log_message "GPU mismatch on attempt $RETRY_COUNT/$MAX_RETRIES. Max retries reached for $BATCH_CONFIG. Giving up."
        exit 1
    fi
fi

# Any other non-zero exit code: treat as transient failure and retry
if [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    log_message "Transient failure (exit $EXIT_CODE) on attempt $RETRY_COUNT/$MAX_RETRIES. Resubmitting $BATCH_CONFIG (attempt $NEXT_RETRY)..."

    RETRY_DELAY=$(( 1800 + RANDOM % 1801 ))
    log_message "Waiting ${RETRY_DELAY}s before resubmitting..."
    sleep "$RETRY_DELAY"

    job_output=$($SBATCH_CMD --job-name="${MODEL_SLUG}" --export=ALL,EXPECTED_GPU="$EXPECTED_GPU",CONFIG_BASENAME="$CONFIG_BASENAME" "$SCRIPT_DIR/run_benchmark_slurm.submit" "$BATCH_CONFIG")
    if [ $? -ne 0 ]; then
        log_message "FAILED to resubmit $BATCH_CONFIG on retry $NEXT_RETRY"
        exit 1
    fi

    new_job_id=$(echo "$job_output" | awk '{print $4}')
    log_message "Resubmitted $BATCH_CONFIG as job $new_job_id (attempt $NEXT_RETRY/$MAX_RETRIES)"

    $CLEANUP_SBATCH_CMD \
        --dependency=afterany:"$new_job_id" \
        --job-name="retry_$(basename "$BATCH_CONFIG" .yaml)" \
        --output=/dev/null --error=/dev/null \
        --wrap="$SCRIPT_DIR/retry_on_gpu_mismatch.sh $new_job_id $BATCH_CONFIG $NEXT_RETRY $STATE_FILE"

    exit 0
else
    log_message "Transient failure (exit $EXIT_CODE) on attempt $RETRY_COUNT/$MAX_RETRIES. Max retries reached for $BATCH_CONFIG. Giving up."
    exit 1
fi
