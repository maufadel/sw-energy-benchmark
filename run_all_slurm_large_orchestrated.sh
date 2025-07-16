#!/bin/bash

# --- Configuration ---
MAIN_CONFIG="config-150.yaml"
BATCH_SIZE=10
TEMP_CONFIG_DIR="temp_configs"
MAX_JOBS=10
# Number of jobs submitted per batch config
JOBS_PER_BATCH=5


# --- 1. Run the Orchestrator ---
# This script will read the main config and create smaller, temporary config files.
echo "Running the Slurm orchestrator to generate batch configs..."
python slurm_orchestrator.py "$MAIN_CONFIG" "$BATCH_SIZE"

# Check if the orchestrator ran successfully
if [ $? -ne 0 ]; then
    echo "Orchestrator script failed. Exiting."
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
    sbatch --partition=gpu_top --nodelist=fresko --gres=gpu:l40spcie:1 --time=1440 ./run_benchmark_slurm.submit "$batch_config"

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
