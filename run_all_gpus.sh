#!/bin/bash
#
# Run run_train_test_model.py across all available GPUs.
#
# How it works:
#   1. Detect the number of GPUs (auto, respects CUDA_VISIBLE_DEVICES; override
#      with the NUM_GPUS env var).
#   2. Generate the hyperparameter config files ONCE (prepare_configs.py).
#   3. Launch one worker per GPU. Config i is handled by the worker whose
#      device_id == i % num_workers (the sharding lives in run_train_test_model.py),
#      so passing --num_gpu is what actually distributes the work.
#   4. Wait for every worker and report any failures.
#   5. Merge the per-GPU result logs into single config-ordered files.
#
# Each worker writes its own output/<exp>/val_results_<gpu>.txt and
# loss_log_<gpu>.txt; stdout/stderr go to output/<exp>/gpu_<gpu>.log.
#
set -uo pipefail

# ============================================
# Configuration -- edit for your run
# ============================================
META_CONFIG="./meta_config/meta_config_common_432_new.json"
EXPERIMENT_NAME="new_dataset"
INPUT_SCALING_METHOD="standard"
TARGET_SCALING_METHOD="standard"
IS_GRAPH_LEVEL="true"
IS_CONTINUOUS="true"     # true => skip configs already recorded in val_results_<gpu>.txt
VERBOSE="True"
SEED=42
FOLD_NAME=""             # leave empty unless using a CV fold

# ============================================
# Validate
# ============================================
if [[ ! -f "$META_CONFIG" ]]; then
    echo "Error: meta config not found: $META_CONFIG" >&2
    exit 1
fi

# ============================================
# Detect number of GPUs (override with NUM_GPUS=...)
# ============================================
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "No CUDA GPUs detected; running a single process (CPU or GPU 0)."
    NUM_GPUS=1
fi

# ============================================
# Generate configs once, then size the worker pool to the work available
# ============================================
python prepare_configs.py --meta_config "$META_CONFIG" --experiment_name "$EXPERIMENT_NAME" || exit 1
CONFIG_DIR="config/${EXPERIMENT_NAME}"
NUM_CONFIGS="$(find "$CONFIG_DIR" -maxdepth 1 -name '*.json' | wc -l | tr -d '[:space:]')"

NUM_WORKERS="$NUM_GPUS"
if [[ "$NUM_CONFIGS" -lt "$NUM_WORKERS" ]]; then
    NUM_WORKERS="$NUM_CONFIGS"   # no point launching idle workers
fi

OUTPUT_DIR="output/${EXPERIMENT_NAME}"
[[ -n "$FOLD_NAME" ]] && OUTPUT_DIR="${OUTPUT_DIR}/${FOLD_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  Experiment : $EXPERIMENT_NAME"
echo "  Meta config: $META_CONFIG"
echo "  GPUs       : $NUM_GPUS"
echo "  Configs    : $NUM_CONFIGS"
echo "  Workers    : $NUM_WORKERS"
echo "=========================================="
echo

# ============================================
# Build the python args for a given device id
# ============================================
build_args() {
    local device_id=$1
    local args=(
        --meta_config "$META_CONFIG"
        --experiment_name "$EXPERIMENT_NAME"
        --device_id "$device_id"
        --num_gpu "$NUM_WORKERS"
        --skip_config_generation true
        --input_scaling_method "$INPUT_SCALING_METHOD"
        --target_scaling_method "$TARGET_SCALING_METHOD"
        --is_graph_level "$IS_GRAPH_LEVEL"
        --is_continuous "$IS_CONTINUOUS"
        --verbose "$VERBOSE"
        --seed "$SEED"
    )
    [[ -n "$FOLD_NAME" ]] && args+=(--fold_name "$FOLD_NAME")
    printf '%s\n' "${args[@]}"
}

# ============================================
# Launch one worker per GPU
# ============================================
pids=()
for ((i = 0; i < NUM_WORKERS; i++)); do
    mapfile -t worker_args < <(build_args "$i")
    logfile="${OUTPUT_DIR}/gpu_${i}.log"
    echo "Starting worker on GPU $i  ->  $logfile"
    python run_train_test_model.py "${worker_args[@]}" > "$logfile" 2>&1 &
    pids+=($!)
done
echo

# ============================================
# Wait for all workers; report failures
# ============================================
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Worker on GPU $i FAILED (see ${OUTPUT_DIR}/gpu_${i}.log)" >&2
        fail=1
    fi
done

# ============================================
# Merge per-GPU result logs (even if some workers failed)
# ============================================
merge_args=(--experiment_name "$EXPERIMENT_NAME")
[[ -n "$FOLD_NAME" ]] && merge_args+=(--fold_name "$FOLD_NAME")
python merge_gpu_logs.py "${merge_args[@]}"

echo
if [[ "$fail" -eq 0 ]]; then
    echo "All workers finished successfully. Merged results in $OUTPUT_DIR."
else
    echo "Finished with failures; merged available results in $OUTPUT_DIR." >&2
fi
exit "$fail"
