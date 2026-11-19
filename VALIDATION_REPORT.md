# GemmaCode E2B - Phase 1 Validation (Static)

This file captures **static validation performed in the local workspace**. Runtime validation (model download/load, inference, GPU benchmarking) must be executed in **Google Colab** and is not claimed here.

## Verified technical references (official docs reviewed)
- Hugging Face model card: https://huggingface.co/google/gemma-4-E2B-it
- Transformers Gemma 4 docs: https://huggingface.co/docs/transformers/model_doc/gemma4
- Google AI Gemma inference docs: https://ai.google.dev/gemma/docs/core/huggingface_inference
- Google AI thinking docs: https://ai.google.dev/gemma/docs/capabilities/thinking

### Key verified facts used in implementation
- Model ID: `google/gemma-4-E2B-it` (instruction-tuned Gemma 4 E2B).
- Multimodal model class/processor: `AutoModelForMultimodalLM` (also referred as image-text-to-text in some docs) + `AutoProcessor`.
- Transformers requirement indicated by official docs: `transformers>=5.10.1`.
- Chat/generation flow: build messages -> `processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=...)` -> `model.generate(...)` -> decode with `skip_special_tokens=False` -> `processor.parse_response(...)` for structured parsing.
- `enable_thinking` is supported via `apply_chat_template(..., enable_thinking=True|False)`. E2B behavior differs from larger models when thinking is disabled (template/model behavior should be followed, not hardcoded).
- Access: Gemma models on Hugging Face may be gated; users must accept terms and authenticate (Colab Secrets/HF token) before download in Colab.

## Workspace/static checks completed
- Project repository structure created for Phase 1 (configs, data dirs, notebooks, src packages, benchmarks, outputs, deployment placeholders).
- `benchmarks/coding_tasks.jsonl` created with 5 required software-engineering tasks.
- `notebooks/01_baseline.ipynb` authored with required sections: environment diagnostics, dependency installation (baseline inference), HF auth via Colab Secrets, model loading, `ask_gemma()` helper, benchmark execution, and result saving to local + Google Drive with non-overwriting run handling.
- Evaluation foundation implemented (`src/evaluation`: env diagnostics, rubric definitions, JSON/JSONL I/O helpers).
- Placeholder notebooks created for future phases: `02_dataset_preparation.ipynb`, `03_qlora_experiment.ipynb`, `04_evaluation.ipynb`.
- `.gitignore` configured to exclude caches, secrets, datasets, and model/adapter/checkpoint weights.
- Requirements and config files added for baseline inference and documented future training/evaluation settings.

## Static validation commands/results (local)
- Notebook JSON validity: **TO BE RECORDED AFTER CHECK** (run `python -m json.tool` or equivalent on all `.ipynb`).
- Python syntax checks: **TO BE RECORDED AFTER CHECK** (`python -m py_compile` on implemented modules).
- Dependency/runtime execution: **NOT EXECUTED LOCALLY** (no model downloads; Colab-only inference/training policy).

## Requires Google Colab runtime validation
The following are intentionally **not claimed** as completed in this environment:
- Actual installation/runtime resolution of dependencies in Colab
- Hugging Face gated access success for the executing account
- Download/loading of `google/gemma-4-E2B-it` weights (must occur in Colab, not local laptop)
- GPU diagnostics on Colab hardware (T4/L4/A100 etc.)
- Successful execution of `ask_gemma()` and all 5 benchmark prompts
- Generation timing/token metrics from real runs
- Saving outputs to mounted Google Drive path `MyDrive/GemmaCode-E2B/benchmarks/results/`

## Safety/compliance notes
- No model weights downloaded to local machine.
- No credentials/tokens hardcoded.
- Fine-tuning (Phase 2+) not started.
