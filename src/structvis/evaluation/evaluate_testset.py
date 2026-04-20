from argparse import ArgumentParser
from collections import defaultdict
from statistics import mean

from datasets import load_dataset
from src.util import check_reasoning, check_reasoning_code, extract_part, save_json
from vllm import LLM, SamplingParams

TEMPERATURE = 0.2
TOP_P = 0.95
CONTEXT_GENERATE = 4096


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output file.")
    parser.add_argument("--is_reasoning_model", action="store_true", help="Flag indicating if the model is a reasoning model.")
    parser.add_argument("--use_think_tag", action="store_true", help="Flag indicating if the think tag is added to the system prompt.")
    args = parser.parse_args()
    return args


def compute_statistics(data):
    correct_flags = [sample["result"]["correct"] for sample in data]
    total_correct = sum(correct_flags)
    mean_correct = mean(correct_flags) if correct_flags else 0

    # Per difficulty
    difficulty_stats = defaultdict(list)
    for sample in data:
        difficulty_stats[sample["difficulty"]].append(sample["result"]["correct"])

    difficulty_results = {
        diff: {"total": len(vals), "count": sum(vals), "mean": mean(vals) if vals else 0} for diff, vals in difficulty_stats.items()
    }

    # Per category
    cat_stats = defaultdict(list)
    for sample in data:
        cat_stats[sample["category_key"]].append(sample["result"]["correct"])

    cat_results = {subj: {"total": len(vals), "count": sum(vals), "mean": mean(vals) if vals else 0} for subj, vals in cat_stats.items()}

    # Per type
    type_stats = defaultdict(list)
    for sample in data:
        type_stats[sample["type"]].append(sample["result"]["correct"])

    type_results = {ty: {"total": len(vals), "count": sum(vals), "mean": mean(vals) if vals else 0} for ty, vals in type_stats.items()}

    # Final statistics dictionary
    stats = {
        "total": {"samples": len(data), "correct_count": total_correct, "correct_mean": mean_correct},
        "per_type": type_results,
        "per_category": cat_results,
        "per_difficulty": difficulty_results,
    }

    return stats


if __name__ == "__main__":
    dataset = load_dataset(path="anonymized/StructVisTestset", split="test")

    args = get_args()
    save_filepath = args.output_path
    is_reasoning = args.is_reasoning_model
    model_name = args.model_name_or_path
    use_think_tag = args.use_think_tag
    system_prompt = "You are an AI assistant expert specialized in understanding and interpreting visualizations. Your task is to analyze the provided structured image and respond to queries with correct answers."

    if is_reasoning:
        print("USE REASONING CONTEXT LENGTH")
        CONTEXT_GENERATE *= 2

    print("Loading Model ...")
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=CONTEXT_GENERATE, skip_special_tokens=False)

    if "qwen2.5" in model_name.lower():
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=CONTEXT_GENERATE,
            max_num_seqs=256,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"use_fast": True, "min_pixels": 64 * 28 * 28, "max_pixels": 512 * 28 * 28},
        )
    else:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=CONTEXT_GENERATE,
            max_num_seqs=256,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )

    prompts = []
    outputs = []

    print("Building prompts ...")
    for sample in dataset:
        prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample["user_prompt"]}, {"type": "image_pil", "image_pil": sample["image"]}],
                },
            ]
        )
    print("Run generation ...")
    print(f"LEN prompts: {len(prompts)}")
    outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
    print(f"LEN outputs: {len(outputs)}")

    print("Processing output ...")
    cnt_not_stop = 0
    dataset_final = []
    for data, out in zip(dataset, outputs):
        sample = data
        correct = False
        out_raw = out.outputs[0].text
        out_no_reas = check_reasoning(check_reasoning_code(out_raw))
        answer = extract_part(out_no_reas, "Answer:", "Reason:", True).strip()
        if answer != "":
            if data["answer"].lower() in answer.split("\n")[0].split(")")[0].split(":")[0].lower():
                correct = True
        else:
            if data["answer"].lower() in out_no_reas.split("\n")[0].split(")")[0].split(":")[0].lower():
                correct = True
        sample["result"] = {
            "prompt": out.prompt,
            "generation": out_raw,
            "generation_split": out_no_reas,
            "stop": out.outputs[0].finish_reason,
            "answer": answer,
            "reason": extract_part(out_raw, "Reason:", "", False),
            "correct": correct,
        }
        del sample["image"]
        dataset_final.append(sample)
        if out.outputs[0].finish_reason != "stop":
            cnt_not_stop += 1

    print(f"Finish reason not stop: {cnt_not_stop}.")
    results = compute_statistics(dataset_final)
    results["cnt_not_stop"] = cnt_not_stop
    save_json(filename=f"{save_filepath}.json", data=dataset_final)
    save_json(filename=f"{save_filepath}_result.json", data=results)
