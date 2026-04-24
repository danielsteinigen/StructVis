set -x
set +e

MODEL_NAME=smol-2b-structvis
MODEL_PATH=/models/structvis/$MODEL_NAME
OUTPUT_PATH=/models/envs/eval/logs
TASKS=chartqa,infovqa_val,docvqa_val
BATCH_SIZE=256
DATA_PARALLEL_SIZE=4
MAX_MODEL_LEN=8192
MAX_TOKENS=4096
WANDB_PROJECT=structvis-eval

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)
            MODEL_NAME="$2"
            MODEL_PATH=/models/structvis/$MODEL_NAME
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            MODEL_NAME="$(basename "$MODEL_PATH")"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-parallel-size)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: evaluate_public_bench.sh [--model-name NAME | --model-path PATH] [--output-path PATH] [--tasks TASKS] [--batch-size N] [--data-parallel-size N] [--max-model-len N] [--max-tokens N] [--wandb-project NAME]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

python3 -m lmms_eval \
    --model vllm \
    --model_args model=$MODEL_PATH,data_parallel_size=$DATA_PARALLEL_SIZE,max_model_len=$MAX_MODEL_LEN \
    --gen_kwargs "max_tokens=$MAX_TOKENS,max_new_tokens=$MAX_TOKENS" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path $OUTPUT_PATH \
    --timezone Europe/Berlin \
    --verbosity DEBUG \
    --wandb_args project=$WANDB_PROJECT,name=$MODEL_NAME
