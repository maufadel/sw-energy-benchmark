#!/bin/bash
set -e

# Print the current date and time
echo "[$(date)] Starting"

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
lscpu >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log nvidia-smi
echo "nvidia-smi" >> "$SYSTEM_INFO_FILE"
nvidia-smi >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log free -h
echo "free -h" >> "$SYSTEM_INFO_FILE"
free -h >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# Run and log df -h
echo "df -h" >> "$SYSTEM_INFO_FILE"
df -h >> "$SYSTEM_INFO_FILE"
echo -e "\n" >> "$SYSTEM_INFO_FILE"

# ---------------------------------------
# Install Python virtual environment.
echo "[$(date)] Creating Python virtual environment..."
uv venv --python python3.11 .venv
echo "[$(date)] Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "[$(date)] Installing requirements..."
uv pip install -r requirements.txt

# ---------------------------------------
# Run the Python scripts for language
echo "[$(date)] Running llm_server_optimized.py..."
python language/llm_server_optimized.py "$RESULTS_DIR"
#echo "[$(date)] Running llm_batch_vanilla.py..."
#python language/llm_batch_vanilla.py "$RESULTS_DIR"
echo "[$(date)] Running llm_batch_optimized.py..."
python language/llm_batch_optimized.py "$RESULTS_DIR"

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

# Deactivate the virtual environment at the end
echo "[$(date)] Deactivating virtual environment..."
deactivate

# Print the current date and time
echo "[$(date)] Finished"
