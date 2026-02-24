#!/bin/bash
# reserve_gpu.sh
#
# Submit a SLURM job that occupies a GPU of the specified type for a given
# duration. Useful for keeping the "wrong" H100 variant busy so that benchmark
# retries land on the correct one.
#
# For H100_SXM / H100_PCIe (which share the same GRES label), the job verifies
# the actual GPU via nvidia-smi and releases immediately if it's the wrong
# variant — only the correct type is held for the full duration.
#
# Usage:
#   ./reserve_gpu.sh --gpu <type> --time <minutes> [--account <name>] [--count <n>]

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./reserve_gpu.sh --gpu <type> --time <minutes> [options]

Required:
  --gpu <type>      GPU type: V100, A100_SXM, A100_PCIe, H100_SXM, H100_PCIe, H200, L40S
  --time <minutes>  How long to hold the GPU (in minutes)

Optional:
  --account <name>  SLURM account (default: init)
  --count <n>       Number of GPUs to reserve simultaneously (default: 1)
  -h, --help        Show this help

Notes:
  For H100_SXM and H100_PCIe, the submitted job checks the actual GPU variant
  via nvidia-smi and exits immediately if it received the wrong one. Submit
  --count jobs to increase the chance of hitting the right variant.

Examples:
  # Block one H100 SXM/NVL so H100 PCIe benchmark retries get through:
  ./reserve_gpu.sh --gpu H100_SXM --time 120

  # Block two H100 PCIe slots for 90 minutes:
  ./reserve_gpu.sh --gpu H100_PCIe --time 90 --count 2

  # Reserve an A100 PCIe for 30 minutes under a different account:
  ./reserve_gpu.sh --gpu A100_PCIe --time 30 --account myproject
EOF
}

# --- Defaults ---
GPU_TYPE=""
TIME_MINUTES=""
ACCOUNT="init"
COUNT=1

# --- Argument Parsing ---
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)     GPU_TYPE="$2";     shift 2 ;;
        --time)    TIME_MINUTES="$2"; shift 2 ;;
        --account) ACCOUNT="$2";      shift 2 ;;
        --count)   COUNT="$2";        shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown option '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

# --- Validation ---
[ -z "$GPU_TYPE" ]     && { echo "Error: --gpu is required."  >&2; usage >&2; exit 1; }
[ -z "$TIME_MINUTES" ] && { echo "Error: --time is required." >&2; usage >&2; exit 1; }

if ! [[ "$TIME_MINUTES" =~ ^[0-9]+$ ]] || [ "$TIME_MINUTES" -eq 0 ]; then
    echo "Error: --time must be a positive integer (minutes)." >&2; exit 1
fi
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -eq 0 ]; then
    echo "Error: --count must be a positive integer." >&2; exit 1
fi

# --- GPU Mapping (mirrors run_benchmark_gpu.sh) ---
EXPECTED_GPU=""
case "$GPU_TYPE" in
    V100)
        PARTITION="gpu";     GRES="gpu:v100sxm:1";   NODES="losangeles,sanfrancisco,sandiego" ;;
    A100_SXM)
        PARTITION="gpu";     GRES="gpu:a100sxm:1";   NODES="sacramento" ;;
    A100_PCIe)
        PARTITION="gpu_top"; GRES="gpu:a100pcie:1";  NODES="fresko" ;;
    H100_SXM)
        PARTITION="gpu_top"; GRES="gpu:h100pcie:1";  NODES="sanjose"; EXPECTED_GPU="H100_SXM" ;;
    H100_PCIe)
        PARTITION="gpu_top"; GRES="gpu:h100pcie:1";  NODES="sanjose"; EXPECTED_GPU="H100_PCIe" ;;
    H200)
        PARTITION="gpu_top"; GRES="gpu:h200sxm:1";   NODES="trinity" ;;
    L40S)
        PARTITION="gpu_top"; GRES="gpu:l40spcie:1";  NODES="fresko" ;;
    *)
        echo "Error: Unknown GPU type '$GPU_TYPE'." >&2
        echo "Valid types: V100, A100_SXM, A100_PCIe, H100_SXM, H100_PCIe, H200, L40S" >&2
        exit 1 ;;
esac

SLEEP_SECS=$(( TIME_MINUTES * 60 ))

# --- Build job script in a temp file ---
# SLURM reads the script at submission time, so the temp file can be removed
# immediately after sbatch returns.
TMPSCRIPT=$(mktemp /tmp/reserve_gpu_XXXXXX.sh)
trap 'rm -f "$TMPSCRIPT"' EXIT

cat > "$TMPSCRIPT" <<JOBSCRIPT
#!/bin/bash
GPU_NAME=\$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits 2>/dev/null | head -n 1)
echo "Detected GPU: \$GPU_NAME"
JOBSCRIPT

# Append GPU-type validation for H100 variants (they share the same GRES label)
if [ "$EXPECTED_GPU" = "H100_SXM" ]; then
    cat >> "$TMPSCRIPT" <<JOBSCRIPT
if echo "\$GPU_NAME" | grep -qi "pcie"; then
    echo "Got H100 PCIe instead of H100 SXM/NVL — releasing immediately."
    exit 0
fi
echo "Confirmed H100 SXM/NVL. Holding for ${TIME_MINUTES} minutes..."
JOBSCRIPT
elif [ "$EXPECTED_GPU" = "H100_PCIe" ]; then
    cat >> "$TMPSCRIPT" <<JOBSCRIPT
if ! echo "\$GPU_NAME" | grep -qi "pcie"; then
    echo "Got H100 SXM/NVL instead of H100 PCIe — releasing immediately."
    exit 0
fi
echo "Confirmed H100 PCIe. Holding for ${TIME_MINUTES} minutes..."
JOBSCRIPT
else
    cat >> "$TMPSCRIPT" <<JOBSCRIPT
echo "Holding ${GPU_TYPE} for ${TIME_MINUTES} minutes..."
JOBSCRIPT
fi

cat >> "$TMPSCRIPT" <<JOBSCRIPT
sleep ${SLEEP_SECS}
echo "Done."
JOBSCRIPT

chmod +x "$TMPSCRIPT"

# --- Submit ---
echo "Reserving ${COUNT}x ${GPU_TYPE} on ${NODES} for ${TIME_MINUTES} min (account: ${ACCOUNT})"
[ -n "$EXPECTED_GPU" ] && echo "Note: each job will release immediately if it gets the wrong H100 variant."
echo ""

for (( i=1; i<=COUNT; i++ )); do
    JOB_ID=$(sbatch \
        --parsable \
        --partition="$PARTITION" \
        --nodelist="$NODES" \
        --gres="$GRES" \
        --time="$TIME_MINUTES" \
        --account="$ACCOUNT" \
        --ntasks=1 \
        --nodes=1 \
        --job-name="hold_${GPU_TYPE}" \
        --output="/dev/null" \
        --error="/dev/null" \
        "$TMPSCRIPT")
    echo "  [${i}/${COUNT}] Submitted job ${JOB_ID}  →  scancel ${JOB_ID}"
done

echo ""
echo "Monitor: squeue -u \$USER --format='%.10i %.12j %.8T %.10M %.10l %N'"
