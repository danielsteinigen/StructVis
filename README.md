# StructVis

StructVis provides a comprehensive framework for generating and interpreting **Struct**ured **Vis**ualizations through Formal Representation Languages (FRLs) for Multi-Domain Vision-Language Understanding.

## Features
- 🧱 scalable domain-agnostic pipeline for synthetically generating structured diagram datasets utilizing FRL code using the [Structivize](https://github.com/danielsteinigen/structivize) rendering toolkit
- 🎚️ Controlled structural complexity and diversity via levels and persona-driven prompts to reflect real-world domain problems
- 🔗 Context-rich samples with explicit code-to-image mapping and problem-solution pairs
- 🌍 Coverage of 47 FRLs, 28 visualization types, and 7 domains
- ✅ Multi-stage quality filtering (deduplication, correctness checks, node statistics, proportions, image variance)
- 🧾 QA refinement pipeline with 8 question types (closed-ended and open-ended)
- 🧪 LLM evaluation with respect to their ability to generate code in distinct domain-specific FRLs in terms of code validity and complexity
- 🧠 VLM fine-tuning and evaluation with with a training paradigm that incorporates the code representation of the image into the model's reasoning trace, enabling the VLM to internalize and utilize a symbolic intermediate space

## Publications
- **Code-Guided Reasoning in Vision-Language Models for Complex Diagram Understanding** — ESANN 2026. [DOI](https://doi.org/10.14428/esann/2026.ES2026-372)

## Datasets & Models
- 📦 StructVis Dataset: (add link)
- 📦 Personas Dataset: (add link)
- 🤖 StructVis Model: (add link)

## Installation

Create virtual Python environment e.g. using uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
```

Install dependencies via optional groups (recommended):
```bash
uv pip install -e ".[datagen]"
uv pip install -e ".[train]"
uv pip install -e ".[eval]"
uv pip install -e ".[all]"
```

Alternative (legacy):
```bash
uv pip install -r requirements.txt
```

Install Structivize (required for rendering). Follow the setup in the Structivize repo:
https://github.com/danielsteinigen/structivize

Download required NLTK data (for evaluation):
```bash
python -c "import nltk; nltk.download('wordnet')"
```

## Usage

## Data generation

### Persona generation
Generate persona descriptions from FineWebEdu:
```
python structvis/src/data_generator/persona_data_generator.py --config src/data_generator/configs/llm/qwen-3-235b-instruct.yaml --output personas.jsonl --data-batch-size 50000
```

Generate search queries to find appropriate personas for each image category. Put path to input file in `input_path`:
```
python structvis/src/data_generator/persona_query_data_generator.py --config src/data_generator/configs/qwen-3-235b-instruct.yaml --output personas_query.jsonl --data-batch-size 50000
```

Filter the persona dataset to get 25,000 personas per image category:
```
python structvis/src/personas/filter_personas_semantic_hf.py --input personas.jsonl --output personas_filtered --query-path data/categories_all.json
```

### StrcutVis data generation
Generate StructVis dataset using the personas as input. Put path to input file in `input_path`:
```
python structvis/src/data_generator/structured_data_generator.py --config src/data_generator/configs/llm/qwen-3-coder-480b-instruct.yaml --output structvis_generations.jsonl --data-batch-size=50000
```

#### Rendering
Render images for each generated sample. Put path to generated dataset in `data_files`:
```
python toolkit/src/structivize/render_batch.py
```

### Filtering
Filter the generated vision-langauge dataset according to diffferent criteria. Set `save_dirs` and `save_filepath`:
```
python structvis/src/structvis_dataset/filter_generations.py
```

Perform LLM-as-a-Judge scoring to retrieve high quality samples. Put path to input file in `input_path`:
```
python structvis/src/data_generator/scoring_data_generator.py --config=src/data_generator/configs/llm/gpt-oss-120b.yaml --output=dataset_score.jsonl --data-batch-size=50000
```

Split the dataset into subgroups for generating different question types. Set `save_dirs` and `save_filepath`:
```
python structvis/src/structvis_dataset/split_dataset.py
```


### Refinement
Generate closed-ended questions for a subset of the dataset. Put path to input file in `input_path`:
```
python structvis/src/data_generator/qa_data_generator.py --config=src/data_generator/configs/llm/qwen-3-coder-480b-instruct.yaml --output=dataset_cec.jsonl --data-batch-size=50000
```

Generate captions for a subset of the dataset. Put path to input file in `input_path`:
```
python structvis/src/data_generator/caption_data_generator.py --config=src/data_generator/configs/llm/qwen-3-coder-480b-instruct.yaml --output=dataset_captions.jsonl --data-batch-size=50000
```

Assemble the subsets of the different question types into a single training dataset. Set `save_filepath`:
```
python structvis/src/structvis_dataset/assemble_dataset.py
```

## Training
Fine-tune a VLM on the training dataset. The hyperparameter can be configured at the beginning of the script:
```
python structvis/src/training/train_sft.py
```

## Evaluation
Evaluate the perfomance of LLMs in generating code in specific FRLs. Set `save_dirs` to the directories of the rendering outputs:
```
python structvis/src/evaluation/evaluate_code_generation.py
```

Evaluate the performance of the trained models on the StructVis testset:
```
python structvis/src/benchmark/evaluate_testset.py --model_name_or_path "/models/structvis/smolvlm2-2b-checkpoint-3400" --output_path "smolvlm2-2b-3400-test"
```

Evaluate the performance of the trained models on public benchmarks. Set path to model:
```
./structvis/src/evaluation/evaluate_public_bench.sh
```
