#!/bin/bash
set -e

# --- Configuration ---
# Use first argument as config file, default to config.yaml
MAIN_CONFIG="${1:-config.yaml}"
# Extract config basename without extension and path
CONFIG_BASENAME=$(basename "$MAIN_CONFIG" .yaml)
TEMP_CONFIG_DIR="temp_configs_${CONFIG_BASENAME}"
MIN_DISK_SPACE_GB=80
# Default Hugging Face cache directory, can be overridden by environment variable
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

# --- Load .env file for HF_TOKEN ---
if [ -f .env ]; then
    echo "[$(date)] Loading .env file..."
    set -a
    source .env
    set +a
else
    echo "[$(date)] Warning: .env file not found."
fi

# --- Check if HF_TOKEN is available ---
if [ -z "$HF_TOKEN" ]; then
    echo "[$(date)] ERROR: HF_TOKEN is not set!"
    echo "[$(date)] Please set HF_TOKEN either:"
    echo "  1. In a .env file in the current directory, or"
    echo "  2. As an environment variable: export HF_TOKEN=your_token"
    echo "[$(date)] You can get a token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "[$(date)] HF_TOKEN is available (length: ${#HF_TOKEN} characters)"

# --- Function to check disk space and clean cache ---
manage_hf_cache() {
    echo "[$(date)] Checking available disk space..."
    # Get available space in GB. df -k is POSIX compliant.
    AVAILABLE_SPACE_GB=$(df -k . | awk 'NR==2 {print int($4 / 1024 / 1024)}')

    echo "[$(date)] Available space: $AVAILABLE_SPACE_GB GB"

    if [ "$AVAILABLE_SPACE_GB" -lt "$MIN_DISK_SPACE_GB" ]; then
        echo "[$(date)] Low disk space detected. Available: $AVAILABLE_SPACE_GB GB, Required: $MIN_DISK_SPACE_GB GB."
        echo "[$(date)] Attempting to clean Hugging Face cache at $HF_CACHE_DIR/hub..."

        if [ ! -d "$HF_CACHE_DIR/hub" ]; then
            echo "[$(date)] Hugging Face cache directory not found. Nothing to clean."
            return
        fi

        # Find and delete oldest model directories until space is sufficient
        while [ "$(df -k . | awk 'NR==2 {print int($4 / 1024 / 1024)}')" -lt "$MIN_DISK_SPACE_GB" ]; do
            # Find the oldest model directory based on modification time
            OLDEST_MODEL_DIR=$(find "$HF_CACHE_DIR/hub" -maxdepth 1 -type d -name 'models--*' -printf '%T@ %p\n' | sort -n | head -n 1 | cut -d' ' -f2-)

            if [ -z "$OLDEST_MODEL_DIR" ]; then
                echo "[$(date)] No more models to delete from cache, but disk space is still low."
                echo "[$(date)] WARNING: Could not free up enough space. Continuing anyway."
                break
            fi

            echo "[$(date)] Deleting oldest model cache directory to free up space: $OLDEST_MODEL_DIR"
            du -sh "$OLDEST_MODEL_DIR"
            rm -rf "$OLDEST_MODEL_DIR"
            echo "[$(date)] Deleted."
            
            # Check space again
            CURRENT_SPACE_GB=$(df -k . | awk 'NR==2 {print int($4 / 1024 / 1024)}')
            echo "[$(date)] New available space: $CURRENT_SPACE_GB GB"
        done
    else
        echo "[$(date)] Sufficient disk space available."
    fi
}

# --- Function to Create Individual Model Configs ---
create_individual_configs() {
    local main_config_path=$1
    echo "[$(date)] Creating individual model configs from $main_config_path..."

    if [ ! -f "$main_config_path" ]; then
        echo "[$(date)] Error: Main config file not found at $main_config_path"
        exit 1
    fi

    # Create a directory for temporary configs
    mkdir -p "$TEMP_CONFIG_DIR"

    # Extract the list of models from the main config file.
    mapfile -t all_models < <(sed -n '/^LLM_MODELS:/,/^[^[:space:]]/p' "$main_config_path" | grep '  - ' | sed 's/  - //;s/\r$//')

    if [ ${#all_models[@]} -eq 0 ]; then
        echo "[$(date)] Warning: No 'LLM_MODELS' found in the config file."
        return
    fi

    # Get the parts of the config file before and after the LLM_MODELS block
    config_prefix=$(awk '/^LLM_MODELS:/ {exit} {print}' "$main_config_path")
    config_suffix=$(awk 'f{print} /^[^[:space:]]/ && p {f=1} /^LLM_MODELS:/ {p=1}' "$main_config_path")

    local num_models=${#all_models[@]}
    echo "[$(date)] Splitting $num_models models into individual config files in $TEMP_CONFIG_DIR"

    for (( i=0; i<num_models; i++ )); do
        local model="${all_models[i]}"
        # Sanitize model name for a valid filename
        local model_filename=$(echo "$model" | tr '/' '_' | tr -c 'a-zA-Z0-9_-' '_')
        local temp_config_filename="$TEMP_CONFIG_DIR/config_${model_filename}.yaml"

        {
            echo "$config_prefix"
            echo "LLM_MODELS:"
            echo "  - $model"
            if [ -n "$config_suffix" ]; then
                echo "$config_suffix"
            fi
        } > "$temp_config_filename"
    done
    echo "[$(date)] Finished creating individual model configs."
}

# --- Main Execution ---

# --- 1. Check if config file exists ---
if [ ! -f "$MAIN_CONFIG" ]; then
    echo "[$(date)] Error: Config file not found at $MAIN_CONFIG"
    echo "Usage: $0 [config_file.yaml]"
    echo "  If no config file is specified, defaults to config.yaml"
    exit 1
fi

echo "[$(date)] Using config file: $MAIN_CONFIG"
echo "[$(date)] Temp config directory: $TEMP_CONFIG_DIR"

# --- 2. Generate Batch Configs ---
# Check if resuming from existing temp directory (for manual resume)
if [ -d "$TEMP_CONFIG_DIR" ] && [ -n "$(ls -A "$TEMP_CONFIG_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] Found existing temp directory with configs. Resuming from: $TEMP_CONFIG_DIR"
else
    echo "[$(date)] Starting a new run. Generating batch configs from $MAIN_CONFIG..."
    create_individual_configs "$MAIN_CONFIG"

    if [ ! -d "$TEMP_CONFIG_DIR" ] || [ -z "$(ls -A "$TEMP_CONFIG_DIR")" ]; then
        echo "[$(date)] Batch config generation failed or produced no files. Exiting."
        exit 1
    fi
fi

echo "[$(date)] Starting benchmark run"

# Create a unique results directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/run_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "[$(date)] Results will be stored in: $RESULTS_DIR"

# ---------------------------------------
# Register system info.
echo "[$(date)] Registering system info..."
SYSTEM_INFO_FILE="$RESULTS_DIR/system_info.txt"

# Run and log lscpu
echo "lscpu" >> "$SYSTEM_INFO_FILE"
lscpu >> "$SYSTEM_INFO_FILE" 2>/dev/null || echo "lscpu not found" >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log nvidia-smi
echo "nvidia-smi" >> "$SYSTEM_INFO_FILE"
nvidia-smi >> "$SYSTEM_INFO_FILE" 2>/dev/null || echo "nvidia-smi not found" >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log free -h
echo "free -h" >> "$SYSTEM_INFO_FILE"
free -h >> "$SYSTEM_INFO_FILE" 2>/dev/null || echo "free command not found" >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log df -h
echo "df -h" >> "$SYSTEM_INFO_FILE"
df -h >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# ---------------------------------------
# Install Python virtual environment.
echo "[$(date)] Creating Python virtual environment..."
uv venv --python python3.11 --clear .venv
echo "[$(date)] Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "[$(date)] Installing requirements..."
uv pip install -r requirements_modern.txt

# ---------------------------------------
# Split config and run benchmarks for each model
if [ -z "$(ls -A "$TEMP_CONFIG_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] No temporary configs created. This might mean LLM_MODELS is empty or not found in $MAIN_CONFIG."
    echo "[$(date)] Skipping LLM benchmarks."
else
    echo "[$(date)] Running benchmarks for each LLM model..."
    for temp_config in "$TEMP_CONFIG_DIR"/config_*.yaml; do
        echo "====================================================="
        echo "[$(date)] Processing model config: $temp_config"
        
        # Manage cache before running the script that might download a model
        manage_hf_cache

        # Create a unique subdirectory for this model's results
        model_filename_part=$(basename "$temp_config" .yaml | sed 's/^config_//')
        MODEL_RESULTS_DIR="$RESULTS_DIR/$model_filename_part"
        mkdir -p "$MODEL_RESULTS_DIR"
        echo "[$(date)] Storing results for this model in: $MODEL_RESULTS_DIR"

        # Optional debug flags (uncomment if needed for troubleshooting):
        # export VLLM_TRACE_FUNCTION=1        # Record vLLM function calls
        # export CUDA_LAUNCH_BLOCKING=1       # Make CUDA operations synchronous (helps identify hanging kernels)
        # export NCCL_DEBUG=TRACE             # Detailed NCCL logging for multi-GPU issues

        echo "[$(date)] Running llm_server_optimized.py for $temp_config..."
        # Allow script to continue even if model fails (exit code 1)
        python language/llm_server_optimized.py "$MODEL_RESULTS_DIR" --config "$temp_config" || {
            exit_code=$?
            if [ $exit_code -eq 1 ]; then
                echo "[$(date)] Model failed in server mode but continuing with next model..."
            else
                echo "[$(date)] Unexpected error in server mode (exit code $exit_code), stopping..."
                exit $exit_code
            fi
        }

        echo "[$(date)] Running llm_batch_optimized.py for $temp_config..."
        # Allow script to continue even if model fails (exit code 1)
        python language/llm_batch_optimized.py "$MODEL_RESULTS_DIR" --config "$temp_config" || {
            exit_code=$?
            if [ $exit_code -eq 1 ]; then
                echo "[$(date)] Model failed in batch mode but continuing with next model..."
            else
                echo "[$(date)] Unexpected error in batch mode (exit code $exit_code), stopping..."
                exit $exit_code
            fi
        }

        echo "[$(date)] Finished processing $temp_config"
        rm "$temp_config"
        echo "[$(date)] Removed temporary config: $temp_config"
    done
    echo "====================================================="
fi

# ---------------------------------------
# Run the Python scripts for vision
#echo "[$(date)] Running resnet50_batch.py..."
#python vision/resnet50_batch.py "$RESULTS_DIR"
#echo "[$(date)] Running resnet50_server.py..."
#python vision/resnet50_server.py "$RESULTS_DIR"

# ---------------------------------------
# Run the Python scripts for tabular
#echo "[$(date)] Running tabular_batch.py..."
#python tabular/tabular_batch.py "$RESULTS_DIR"
#echo "[$(date)] Running tabular_server.py..."
#python tabular/tabular_server.py "$RESULTS_DIR"

# ---------------------------------------
# Cleanup
if [ -d "$TEMP_CONFIG_DIR" ] && [ -z "$(ls -A "$TEMP_CONFIG_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] Cleaning up empty temporary config directory: $TEMP_CONFIG_DIR"
    rm -rf "$TEMP_CONFIG_DIR"
fi

# Deactivate the virtual environment at the end
echo "[$(date)] Deactivating virtual environment..."
deactivate

# Print the current date and time
echo "[$(date)] Finished"
