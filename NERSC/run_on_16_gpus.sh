#!/bin/bash

echo "==========================================="
echo "     🚀 NERSC Multi-GPU Task Launcher      "
echo "    Sliding window: Max 16 concurrent jobs "
echo "==========================================="

if [ $# -ne 1 ]; then
    echo "Usage: $0 <command_file>"
    exit 1
fi

CMD_FILE="$1"
if [ ! -f "$CMD_FILE" ]; then
    echo "Error: File '$CMD_FILE' not found."
    exit 1
fi

mkdir -p logs

MAX_PARALLEL=16

declare -a PIDS=()
declare -a GPU_SLOTS=()
declare -a TASK_IDS=()
declare -A STATUS=()

mapfile -t COMMANDS < "$CMD_FILE"
TOTAL_LINES=${#COMMANDS[@]}

CURRENT_LINE=0
launch_task() {
    local line_num=$1
    local cmd="${COMMANDS[$((line_num - 1))]}"
    if [[ -z "$cmd" ]]; then
        return
    fi
    local task_id
    task_id=$(printf "%02d" "$line_num")

    # Find free GPU slot (0..MAX_PARALLEL-1)
    local used=("${GPU_SLOTS[@]}")
    local gpu_slot=""
    for ((i=0; i<MAX_PARALLEL; i++)); do
        if [[ ! " ${used[*]} " =~ " $i " ]]; then
            gpu_slot=$i
            break
        fi
    done
    if [[ -z "$gpu_slot" ]]; then
        echo "No free GPU slot found (this should not happen here)."
        return 1
    fi

    echo "[Task $task_id] → $cmd"
    srun --nodes=1 --ntasks=1 --gpus=1 \
         --gpu-bind=single:$gpu_slot \
         --exclusive --cpu-bind=cores \
         bash -c "echo '[Task $task_id] STARTING'; $cmd; echo '[Task $task_id] DONE'" \
         > "logs/task_$task_id.log" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    GPU_SLOTS+=("$gpu_slot")
    TASK_IDS+=("$task_id")
}

while : ; do
    # Launch new tasks if slots available
    while (( CURRENT_LINE < TOTAL_LINES && ${#PIDS[@]} < MAX_PARALLEL )); do
        ((CURRENT_LINE++))
        launch_task "$CURRENT_LINE"
    done

    if [ "${#PIDS[@]}" -eq 0 ]; then
        # No running jobs, no more tasks to launch
        break
    fi

    # Wait for any job to finish (bash 4.3+)
    wait -n

    # Check which job finished
    for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            wait "${PIDS[i]}"
            exit_code=$?

            task_id="${TASK_IDS[i]}"
            if [ $exit_code -eq 0 ]; then
                STATUS[$task_id]="✅ Success"
            else
                STATUS[$task_id]="❌ Failed (exit code $exit_code)"
            fi

            unset 'PIDS[i]'
            unset 'GPU_SLOTS[i]'
            unset 'TASK_IDS[i]'

            PIDS=("${PIDS[@]}")
            GPU_SLOTS=("${GPU_SLOTS[@]}")
            TASK_IDS=("${TASK_IDS[@]}")
            break
        fi
    done
done

echo
echo "==================== Summary ===================="
for ((i=1; i<=TOTAL_LINES; i++)); do
    tid=$(printf "%02d" "$i")
    status="${STATUS[$tid]}"
    if [ -z "$status" ]; then
        status="❓ Unknown"
    fi
    echo "Task $tid: $status"
done
echo "================================================="
echo "✅ All tasks processed. Logs are in logs/ directory."