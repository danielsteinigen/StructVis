set -x
set +e

MODEL_NAME=smol-2b-structvis
MODEL_PATH=/models/structvis/$MODEL_NAME

python3 -m lmms_eval \
    --model vllm \
    --model_args model=$MODEL_PATH,data_parallel_size=4,max_model_len=8192 \
    --gen_kwargs "max_tokens=4096,max_new_tokens=4096" \
    --tasks chartqa,infovqa_val,docvqa_val \
    --batch_size 256 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path /models/envs/eval/logs \
    --timezone Europe/Berlin \
    --verbosity DEBUG \
    --wandb_args project=structvis-eval,name=$MODEL_NAME
