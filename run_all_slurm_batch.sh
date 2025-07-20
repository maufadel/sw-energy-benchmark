#!/bin/bash

# --- Configuration ---
MAIN_CONFIG="config-50.yaml"
BATCH_SIZE=1
TEMP_CONFIG_DIR="temp_configs"
MAX_JOBS=500
# Number of jobs submitted per batch config
JOBS_PER_BATCH=4

# --- Function to Create Batch Configs in Bash ---
create_batch_configs() {
    local main_config_path=$1
    local batch_size=$2

    if [ ! -f "$main_config_path" ]; then
        echo "Error: Main config file not found at $main_config_path"
        exit 1
    fi

    # Create a directory for temporary configs
    echo "Creating temporary config directory: $TEMP_CONFIG_DIR"
    mkdir -p "$TEMP_CONFIG_DIR"

    # Extract the list of models from the main config file.
    # This reads the lines between 'LLM_MODELS:' and the next unindented line,
    # filters for lines containing '  - ', and cleans up the model name.
    mapfile -t all_models < <(sed -n '/^LLM_MODELS:/,/^[^[:space:]]/p' "$main_config_path" | grep '  - ' | sed 's/  - //;s/\r$//')

    if [ ${#all_models[@]} -eq 0 ]; then
        echo "Warning: No 'LLM_MODELS' found in the config file."
        return
    fi

    # Extract the part of the config *before* the LLM_MODELS line
    config_prefix=$(awk '/^LLM_MODELS:/ {exit} {print}' "$main_config_path")

    # Extract the part of the config *after* the LLM_MODELS block
    # It sets flag 'p' on LLM_MODELS, then sets flag 'f' on the next unindented line, then starts printing.
    config_suffix=$(awk 'f{print} /^[^[:space:]]/ && p {f=1} /^LLM_MODELS:/ {p=1}' "$main_config_path")

    local num_models=${#all_models[@]}
    local num_batches=$(( (num_models + batch_size - 1) / batch_size ))

    echo "Splitting $num_models models into $num_batches batches of (up to) $batch_size models each."

    for (( i=0; i<num_batches; i++ )); do
        local batch_start=$((i * batch_size))
        # Create a slice of the array for the current batch
        local batch_models=("${all_models[@]:batch_start:batch_size}")
        
        #local batch_config_filename="$TEMP_CONFIG_DIR/config_batch_$((i+1)).yaml"
	local batch_config_filename=$(printf "$TEMP_CONFIG_DIR/config_batch_%03d.yaml" $((i+1)))

        # Create the new config file by combining the prefix, the current model batch, and the suffix
        {
            echo "$config_prefix"
            echo "LLM_MODELS:"
            for model in "${batch_models[@]}"; do
                echo "  - $model"
            done
            # Only add suffix if it's not empty
            if [ -n "$config_suffix" ]; then
                echo "$config_suffix"
            fi
        } > "$batch_config_filename"

        echo "  -> Created $batch_config_filename"
    done
}


# --- 1. Generate Batch Configs ---
echo "Generating batch configs from $MAIN_CONFIG..."
create_batch_configs "$MAIN_CONFIG" "$BATCH_SIZE"

# Check if the function created files
if [ ! -d "$TEMP_CONFIG_DIR" ] || [ -z "$(ls -A "$TEMP_CONFIG_DIR")" ]; then
    echo "Batch config generation failed or produced no files. Exiting."
    exit 1
fi


# --- 2. Submit a Slurm Job for Each Batch Config ---
echo "Submitting Slurm jobs for each batch..."

# Find and sort the generated config files to process them in order
for batch_config in $(find "$TEMP_CONFIG_DIR" -name "config_batch_*.yaml" | sort); do
    
    # --- Job Throttling Logic ---
    while true; do
        # Get the number of jobs currently in the queue (running or pending) for the user
        # The '-h' flag on squeue prevents the header from being printed
        current_jobs=$(squeue -u "$USER" -h | wc -l)
        
        # Check if there is room for another batch of jobs
        if [ "$current_jobs" -le $((MAX_JOBS - JOBS_PER_BATCH)) ]; then
            echo "Queue has space ($current_jobs / $MAX_JOBS). Proceeding with next batch."
            break
        else
            echo "Queue is full ($current_jobs / $MAX_JOBS). Waiting for 60 seconds..."
            sleep 60 # Wait for a minute before checking again
        fi
    done

    echo "-----------------------------------------------------"
    echo "Submitting jobs for batch: $batch_config"
    echo "-----------------------------------------------------"
    
    # Submit a job for each partition, just like the original script
    sbatch --partition=gpu --nodelist=sanfrancisco --time=4320 ./run_benchmark_slurm.submit "$batch_config"
    sbatch --partition=gpu --nodelist=sacramento --time=4320 ./run_benchmark_slurm.submit "$batch_config"
    sbatch --partition=gpu_top --nodelist=sanjose --time=1440 ./run_benchmark_slurm.submit "$batch_config"
    sbatch --partition=gpu_top --nodelist=trinity --time=1440 ./run_benchmark_slurm.submit "$batch_config"
    #sbatch --partition=gpu_top --nodelist=fresko --gres=gpu:l40spcie:1 --time=1440 ./run_benchmark_slurm.submit "$batch_config"

done

echo "All batch jobs have been submitted and are being managed by the queue."

# --- 3. Final Check ---
# Wait for all jobs to clear from the queue before finishing the script
while true; do
    current_jobs=$(squeue -u "$USER" -h | wc -l)
    if [ "$current_jobs" -eq 0 ]; then
        echo "All jobs have completed."
        break
    else
        echo "Waiting for the final $current_jobs job(s) to complete..."
        sleep 60
    fi
done


# --- 4. Cleanup (Optional) ---
# You can uncomment the line below to automatically remove the temporary configs after submission.
# echo "Cleaning up temporary config files..."
# rm -rf "$TEMP_CONFIG_DIR"

echo "Orchestration complete."
