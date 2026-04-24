import argparse
import random

import torch
import wandb
from datasets import Image, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from trl import SFTConfig, SFTTrainer

from structvis.util import replace_bpmndi

random.seed(42)

model_id = "/home/Shared/align/models/hf/vlm/SmolVLM2-2.2B-Instruct"
output_dir = "/data/models/structvis/smol-2b-structvis"
dataset_id = "anonymized/StructVisAssembled"

number_samples_test = 2500
MODEL_TYPE = "GEMMA"  # SMOLVLM, QWEN25, MISTRAL, GEMMA
USE_THINK_TAGS = True
LEARNING_RATE = 2e-05  # 2e-05 # 2e-04 1.4e-5  5e-05
BATCH_SIZE = 2  #   4    8   2
ACCU_STEPS = 32  # 16    8   32
EPOCHS = 5
FREEZE_VISION = True
CUSTOM_COLLATE_FN = True
USE_LORA = False
ADD_SPECIAL_TOKENS = False
THINK_TAGS = {
    "think_start": "<think>",
    "think_end": "</think>",
    "lang_start": "<think_lang>",
    "lang_end": "</think_lang>",
    "code_start": "<think_code>",
    "code_end": "</think_code>",
    "think_enable": "/code_think",
}
print(torch.cuda.device_count())

system_message_general = [
    "You are an AI assistant expert specialized in understanding and interpreting visualizations. Your task is to analyze the provided structured image and respond to queries with correct answers.",
    "You are an AI assistant specialized in analyzing images. Your task is to interpret the provided structured image and answer user queries with clarity and accuracy.",
    "You are an AI assistant trained to understand and explain images. Carefully examine the given visualization and provide responses that are correct, detailed, and easy to follow.",
    "You are an AI assistant focused on interpreting structured visualizations. Your role is to analyze the visual input and generate accurate insights, explanations, or answers to user questions.",
    "You are an AI assistant expert in image analysis. Study the provided image and respond to user queries by offering precise interpretations and clear reasoning.",
]
system_message = {
    "qa_problem": "You are an AI assistant specialized in analyzing structured visualizations. When users present a visualization and describe an issue or request, your role is to carefully inspect the image, identify the problem, and provide clear, accurate, and helpful solutions or recommendations.",
    "qa_detail": "You are an AI assistant specialized in analyzing structured visualizations. When users present a visualization and ask a question about it, your role is to carefully inspect the image and respond to the query with a correct answer.",
    "description": "You are an AI assistant expert in interpreting structured visualizations. Your task is to examine the given visualization and produce a precise functional description, explaining what the visualization does, how it works, and what information it conveys.",
    "structural": "You are an AI assistant focused on understanding the structure of visualizations. When users provide a visualization, your task is to analyze its components, layout, relationships between elements, and respond to queries with correct answers.",
    "association": "You are an AI assistant skilled in matching textual content with visualizations. When given a visualization and multiple candidate texts, your task is to determine which text best corresponds to the visualization.",
    "consistency": "You are an AI assistant specialized in validating texts against structured visualizations. When provided with both a visualization and a candidate text, your task is to assess whether the text is factually and conceptually consistent with the visualization.",
    "captioning": "You are an AI assistant trained in captioning structured visualizations. Your task is to generate a concise, accurate, and informative caption that summarizes the main message of the visualization in a way that is clear and useful to the intended audience.",
    "code": "You are an AI assistant expert in image to code translation. Your task is to convert the image provided by the user to an accurate corresponding code representation using an appropriate code language.",
}

reason_hints = [
    "Derive the correct answer by reasoning over the code representation of the image.",
    "Base your reasoning and final answer on the code representation of the image.",
    "Use the code form of the image for reasoning, then provide the correct final answer.",
    "Reason with the code representation of the image to produce the correct answer.",
]


def format_data(sample):
    image = sample["image_augment"]
    sys_msg = system_message_general[random.randint(0, 4)] if random.randint(0, 3) != 0 else system_message[sample["type"]]
    msg_assistent = sample["assistant"]
    if USE_THINK_TAGS:
        sys_msg += f" {reason_hints[random.randint(0, 3)]} {THINK_TAGS['think_enable']}"
        code = sample["code"] if sample["lang_key"] != "bpmn" else replace_bpmndi(sample["code"])
        msg_assistent = (
            f'{THINK_TAGS["think_start"]}\n'
            f'{THINK_TAGS["lang_start"]}{sample["lang_name"]}{THINK_TAGS["lang_end"]}\n'
            f'{THINK_TAGS["code_start"]}\n{code}\n{THINK_TAGS["code_end"]}\n'
            f'{THINK_TAGS["think_end"]}\n{msg_assistent}'
        )
    return {
        "images": [image],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_msg}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": sample["user"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": msg_assistent}],
            },
        ],
    }


def process_vision_info_gemma(messages: list[dict]):
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def main():
    global model_id, output_dir, dataset_id, number_samples_test, MODEL_TYPE, USE_THINK_TAGS, LEARNING_RATE
    global BATCH_SIZE, ACCU_STEPS, EPOCHS, FREEZE_VISION, CUSTOM_COLLATE_FN, USE_LORA, ADD_SPECIAL_TOKENS

    parser = argparse.ArgumentParser(description="Fine-tune a VLM on the StructVis assembled dataset.")
    parser.add_argument("--model-id", default=model_id, help="Base Hugging Face model identifier or local path")
    parser.add_argument("--output-dir", default=output_dir, help="Directory where checkpoints are written")
    parser.add_argument("--dataset-id", default=dataset_id, help="Dataset identifier or local dataset path")
    parser.add_argument("--number-samples-test", type=int, default=number_samples_test, help="Number of evaluation samples to use")
    parser.add_argument(
        "--model-type", default=MODEL_TYPE, choices=["SMOLVLM", "QWEN25", "MISTRAL", "GEMMA"], help="Model family configuration"
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Training learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Per-device training batch size")
    parser.add_argument("--accu-steps", type=int, default=ACCU_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--disable-think-tags", action="store_true", help="Disable think-tag augmentation")
    parser.add_argument("--disable-freeze-vision", action="store_true", help="Fine-tune the vision tower as well")
    parser.add_argument("--disable-custom-collate", action="store_true", help="Disable the custom collator")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--add-special-tokens", action="store_true", help="Add think tags as tokenizer special tokens")
    args = parser.parse_args()

    model_id = args.model_id
    output_dir = args.output_dir
    dataset_id = args.dataset_id
    number_samples_test = args.number_samples_test
    MODEL_TYPE = args.model_type
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    ACCU_STEPS = args.accu_steps
    EPOCHS = args.epochs
    USE_THINK_TAGS = not args.disable_think_tags
    FREEZE_VISION = not args.disable_freeze_vision
    CUSTOM_COLLATE_FN = not args.disable_custom_collate
    USE_LORA = args.use_lora
    ADD_SPECIAL_TOKENS = args.add_special_tokens

    wandb.init(
        project="compimg-train",
        name=output_dir.split("/")[-1],
        config={"model": output_dir.split("/")[-1]},
    )

    dataset = load_dataset(path=dataset_id)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"].select(range(number_samples_test))

    config = AutoConfig.from_pretrained(model_id)
    print(f"Use architecture {config.architectures}\n")

    if USE_LORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # bnb_4bit_quant_storage=torch.bfloat16,
        )

    if MODEL_TYPE == "SMOLVLM":
        print("Use SMOLVLM settings.")
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config if USE_LORA else None
        )
        processor = AutoProcessor.from_pretrained(model_id)
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        if FREEZE_VISION:
            # fine-tune LLM only
            for param in model.model.vision_model.parameters():
                param.requires_grad = False

    elif MODEL_TYPE == "MISTRAL":
        print("Use MISTRAL settings.")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # dtype="float32",
            device_map="auto",
            quantization_config=bnb_config if USE_LORA else None,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        if FREEZE_VISION:
            for name, p in model.named_parameters():
                n = name.lower()
                if "vision" in n or "image" in n or "pixel" in n:
                    p.requires_grad = False

    elif MODEL_TYPE == "QWEN25":
        print("Use QWEN25 settings.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config if USE_LORA else None
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=False, min_pixels=64 * 28 * 28, max_pixels=1024 * 28 * 28)
        image_tokens = [151652, 151653, 151655]

        if FREEZE_VISION:
            for param in model.model.visual.parameters():
                param.requires_grad = False

    elif MODEL_TYPE == "GEMMA":
        print("Use GEMMA settings.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # dtype="float32",
            device_map="auto",
            quantization_config=bnb_config if USE_LORA else None,
        )
        processor = AutoProcessor.from_pretrained(model_id)

        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])]
        print("image_tokens")
        print(image_tokens)
        if FREEZE_VISION:
            for param in model.model.vision_tower.parameters():
                param.requires_grad = False

    if ADD_SPECIAL_TOKENS:
        tokenizer = processor.tokenizer
        special_tokens_dict = {"additional_special_tokens": [val for val in THINK_TAGS.values()]}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} special tokens: {processor.tokenizer.special_tokens_map}\n")
        model.resize_token_embeddings(len(processor.tokenizer))

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        samples = [format_data(sample) for sample in examples]

        if MODEL_TYPE == "SMOLVLM":
            texts = [processor.apply_chat_template(example["messages"], add_generation_prompt=False).strip() for example in samples]
        elif MODEL_TYPE in ["MISTRAL", "GEMMA"]:
            texts = [
                processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False).strip()
                for example in samples
            ]
        elif MODEL_TYPE == "QWEN25":
            texts = [processor.apply_chat_template(example["messages"], tokenize=False).strip() for example in samples]

        if isinstance(processor, Qwen2_5_VLProcessor):
            image_inputs = [process_vision_info(example["messages"])[0] for example in samples]  # Qwen
        elif MODEL_TYPE == "GEMMA":
            image_inputs = [process_vision_info_gemma(example["messages"]) for example in samples]
        else:
            image_inputs = [[(img.convert("RGB") if img.mode != "RGB" else img) for img in example["images"]] for example in samples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Ignore the image token index in the loss computation (model specific)
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        if MODEL_TYPE == "GEMMA":
            labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    training_args = SFTConfig(
        output_dir=output_dir,                                          # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
        num_train_epochs=EPOCHS,                                        # Set the number of epochs to train the model.
        per_device_train_batch_size=BATCH_SIZE,                         # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
        per_device_eval_batch_size=BATCH_SIZE,                          # Batch size for evaluation
        gradient_accumulation_steps=ACCU_STEPS,                         # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
        gradient_checkpointing=True,                                    # Enable gradient checkpointing to reduce memory usage during training.
        gradient_checkpointing_kwargs={"use_reentrant": False},         # Set gradient checkpointing to non-reentrant to avoid issues.
        max_length=None,                                                # To avoid truncation that may remove image tokens during training
        # assistant_only_loss=True,                                       # Train on assistant messages only: ensures that loss is computed only on the assistant responses, ignoring user or system messages. default: false (not yet supported for vision-language models)

        optim="adamw_torch_fused",                                      # Use the fused AdamW optimizer for better performance.
        learning_rate=LEARNING_RATE,                                            # Learning rate for training. 2e-04. 2e-05
        warmup_steps=50, # default: 0
        # weight_decay=0.01, # default: .0
        bf16=True,                                                      # Enable bfloat16 precision for training to save memory and speed up computations. default: None
        # tf32=True,                                                      # use tf32 precision. default: None

        logging_steps=200,                                               # log every 10 steps
        eval_steps=425,                                                  # Steps interval for evaluation
        save_strategy="steps",                                          # Save checkpoints at the end of each epoch. default: epoch, steps
        eval_strategy="steps",                                          # Strategy for evaluation
        save_steps=850,                                                  # Steps interval for saving (1 step = batch_size * gradient_accumulation_steps)
        push_to_hub=False,                                              # Automatically push the fine-tuned model to Hugging Face Hub after training.
        report_to="wandb",                                              # Automatically report metrics to tensorboard. alternative: none, trackio
        # dataset_text_field="",                                          # need a dummy field for collator (Name of the column that contains text data in the dataset.) default: 'text'
        dataset_kwargs={"skip_prepare_dataset": True},                  # Skip dataset preparation to handle preprocessing manually with collator.
        remove_unused_columns=False,                                    # Ensure unused columns are not removed in the collator (important for batch processing). Default: True
        # save_safetensors=False
    )

    if USE_LORA:  # based on QLoRA paper
        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,  # 16
            bias="none",
            target_modules=["q_proj", "v_proj"],  # "all-linear"
            task_type="CAUSAL_LM",
            # modules_to_save=["lm_head", "embed_tokens"],
        )

        training_args.max_grad_norm = 0.3
        training_args.warmup_ratio = 0.03
        training_args.lr_scheduler_type = "constant"
        training_args.learning_rate = 2e-4

        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn if CUSTOM_COLLATE_FN else None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config if USE_LORA else None,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
