#!/bin/bash

# --- Configuration ---
MAIN_CONFIG="config-50.yaml"
BATCH_SIZE=1
TEMP_CONFIG_DIR="temp_configs"
MAX_JOBS=500
LOG_FILE="run_all_slurm_batch.log"
# Number of jobs submitted per batch config
JOBS_PER_BATCH=4

# Get the absolute path for the log file
ABSOLUTE_LOG_FILE=$(pwd)/$LOG_FILE

# --- Function to log messages ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ABSOLUTE_LOG_FILE"
}

# --- Function to Create Batch Configs in Bash ---
create_batch_configs() {
    local main_config_path=$1
    local batch_size=$2

    if [ ! -f "$main_config_path" ]; then
        log_message "Error: Main config file not found at $main_config_path"
        exit 1
    fi

    # Create a directory for temporary configs
    log_message "Creating temporary config directory: $TEMP_CONFIG_DIR"
    mkdir -p "$TEMP_CONFIG_DIR"

    # Extract the list of models from the main config file.
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
        local batch_config_filename=$(printf "$TEMP_CONFIG_DIR/config_batch_%03d.yaml" $((i+1)))

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

# --- 0. Initial Log ---
log_message "--- Orchestration Script Started ---"

# --- 1. Handle Arguments and Generate Batch Configs ---
if [ -n "$1" ]; then
    TEMP_CONFIG_DIR=$1
    log_message "Resuming from provided temp directory: $TEMP_CONFIG_DIR"
    if [ ! -d "$TEMP_CONFIG_DIR" ]; then
        log_message "Error: Provided temp directory '$TEMP_CONFIG_DIR' does not exist."
        exit 1
    fi
else
    log_message "Starting a new run. Generating batch configs from $MAIN_CONFIG..."
    create_batch_configs "$MAIN_CONFIG" "$BATCH_SIZE"

    if [ ! -d "$TEMP_CONFIG_DIR" ] || [ -z "$(ls -A "$TEMP_CONFIG_DIR")" ]; then
        log_message "Batch config generation failed or produced no files. Exiting."
        exit 1
    fi
fi

# --- 2. Submit a Slurm Job for Each Batch Config ---
log_message "Submitting Slurm jobs for each batch..."

for batch_config in $(find "$TEMP_CONFIG_DIR" -name "config_batch_*.yaml" | sort); do
    
    while true; do
        current_jobs=$(squeue -u "$USER" -h | wc -l)
        if [ "$current_jobs" -le $((MAX_JOBS - JOBS_PER_BATCH - 1)) ]; then
            log_message "Queue has space ($current_jobs / $MAX_JOBS). Proceeding with next batch."
            break
        else
            log_message "Queue is full ($current_jobs / $MAX_JOBS). Waiting for 60 seconds..."
            sleep 60
        fi
    done

    log_message "-----------------------------------------------------"
    log_message "Submitting jobs for batch: $batch_config"
    log_message "-----------------------------------------------------"
    
    job_ids_str=""
    job_output=$(sbatch --partition=gpu --nodelist=sanfrancisco --time=4320 ./run_benchmark_slurm.submit "$batch_config")
    if [ $? -eq 0 ]; then job_ids_str="$job_ids_str:$(echo "$job_output" | awk '{print $4}')"; else log_message "Failed to submit for sanfrancisco"; fi
    
    job_output=$(sbatch --partition=gpu --nodelist=sacramento --time=4320 ./run_benchmark_slurm.submit "$batch_config")
    if [ $? -eq 0 ]; then job_ids_str="$job_ids_str:$(echo "$job_output" | awk '{print $4}')"; else log_message "Failed to submit for sacramento"; fi

    job_output=$(sbatch --partition=gpu_top --nodelist=sanjose --time=1440 ./run_benchmark_slurm.submit "$batch_config")
    if [ $? -eq 0 ]; then job_ids_str="$job_ids_str:$(echo "$job_output" | awk '{print $4}')"; else log_message "Failed to submit for sanjose"; fi

    job_output=$(sbatch --partition=gpu_top --nodelist=trinity --time=1440 ./run_benchmark_slurm.submit "$batch_config")
    if [ $? -eq 0 ]; then job_ids_str="$job_ids_str:$(echo "$job_output" | awk '{print $4}')"; else log_message "Failed to submit for trinity"; fi
    
    dependency_list=${job_ids_str#:}

    if [ -n "$dependency_list" ]; then
        log_message "SUBMITTED: $batch_config with Job IDs: $dependency_list"
        
        cleanup_command="echo \"[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETED: $batch_config\" >> \"$ABSOLUTE_LOG_FILE\"; rm '$batch_config'"
        
        cleanup_job_id=$(sbatch --dependency=afterany:$dependency_list --job-name="cleanup_$(basename "$batch_config" .yaml)" --output=/dev/null --error=/dev/null --wrap="$cleanup_command")
        log_message "Submitted cleanup job for $batch_config ($cleanup_job_id)"
    else
        log_message "Warning: Failed to submit any jobs for $batch_config. It will not be processed or cleaned up."
    fi

done

log_message "All batch jobs have been submitted and are being managed by the queue."

# --- 3. Final Check ---
while true; do
    current_jobs=$(squeue -u "$USER" -h | wc -l)
    if [ "$current_jobs" -eq 0 ]; then
        log_message "All jobs have completed."
        break
    else
        remaining_configs=$(find "$TEMP_CONFIG_DIR" -name "config_batch_*.yaml" -type f | wc -l | xargs)
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
