#!/bin/bash

# Print the current date and time
echo "[$(date)] Starting"

# Create the "results" folder if it doesn't exist
mkdir -p results

# ---------------------------------------
# Register system info.
echo "[$(date)] Registering system info..."

# Run and log lscpu
echo "lscpu" >> results/system_info.txt
lscpu >> results/system_info.txt
echo -e "\n" >> results/system_info.txt  # Add a newline

# Run and log nvidia-smi
echo "nvidia-smi" >> results/system_info.txt
nvidia-smi >> results/system_info.txt
echo -e "\n" >> results/system_info.txt  # Add a newline

# Run and log free -h
echo "free -h" >> results/system_info.txt
free -h >> results/system_info.txt
echo -e "\n" >> results/system_info.txt  # Add a newline

# Run and log df -h
echo "df -h" >> results/system_info.txt
df -h >> results/system_info.txt
echo -e "\n" >> results/system_info.txt  # Add a newline

# ---------------------------------------
# Install Python virtual environment.
echo "[$(date)] Creating Python virtual environment..."
python3 -m venv venv

echo "[$(date)] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install requirements
echo "[$(date)] Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------
# Run the Python scripts for language
echo "[$(date)] Running llm_server_optimized.py..."
python language/llm_server_optimized.py
echo "[$(date)] Running llm_batch_vanilla.py..."
python language/llm_batch_vanilla.py
echo "[$(date)] Running llm_batch_optimized.py..."
python language/llm_batch_optimized.py

# ---------------------------------------
# Run the Python scripts for vision
echo "[$(date)] Running resnet50_batch.py..."
python vision/resnet50_batch.py
echo "[$(date)] Running resnet50_server.py..."
python vision/resnet50_server.py

# ---------------------------------------
# Run the Python scripts for tabular
echo "[$(date)] Running tabular_batch.py..."
python tabular/tabular_batch.py
echo "[$(date)] Running tabular_server.py..."
python tabular/tabular_server.py

# ---------------------------------------
# Run Energy-Languages benchmark.
echo "[$(date)] Running Energy-Languages benchmark..."

# Run the Python script with the 'measure' argument
#python energy-languages/el_benchmark.py

# Deactivate the virtual environment at the end
echo "[$(date)] Deactivating virtual environment..."
deactivate

# Print the current date and time
echo "[$(date)] Finished"