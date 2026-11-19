# GemmaCode E2B

Phase 1 foundation for **GemmaCode E2B**, a specialized software-engineering coding model based on Google’s **Gemma 4 E2B Instruct** (`google/gemma-4-E2B-it`).

This repository currently implements **Phase 1 only**:
- project scaffolding and configuration
- environment/dependency setup for baseline inference
- a Colab-ready baseline notebook
- initial software-engineering benchmark tasks
- baseline result collection (local + Google Drive)
- evaluation rubric foundation for future comparisons

> **No fine-tuning is performed in Phase 1.**

## 1. Project vision

Build a locally deployable software-engineering assistant specialized in:
- Java and Spring Boot
- C# and ASP.NET Core
- React and TypeScript
- Python
- C++ and Rust
- SQL and database development
- REST APIs and microservices
- Software architecture and design patterns
- Debugging and root-cause analysis
- Code quality, maintainability and testing

Planned long-term path: QLoRA/LoRA fine-tuning on software-engineering data, optional RAG augmentation, GGUF-compatible export for Ollama deployment, and Pi as a coding-agent interface.

## 2. Architecture (current and planned)

### Phase 1 (implemented now)
- **Base model (baseline):** `google/gemma-4-E2B-it` (Hugging Face, gated)
- **Inference stack:** PyTorch + Transformers (`AutoProcessor`, `AutoModelForMultimodalLM`) + Accelerate
- **Execution environment:** Google Colab (GPU runtime) for model loading/inference
- **Benchmarking:** 5 practical software-engineering prompts stored in JSONL, deterministic baseline generation
- **Evaluation foundation:** human rubric dimensions + result schema/helpers (no fabricated scores)

### Planned (future phases, not implemented yet)
- **Dataset prep:** curated JSONL train/validation/test splits
- **Training:** QLoRA/LoRA fine-tuning (Transformers + PEFT + TRL, compatibility to be re-verified in Colab)
- **Evaluation:** compare original vs fine-tuned vs fine-tuned+RAG
- **Export/deployment:** Safetensors adapters initially; GGUF for compatible Ollama deployment later; Pi integration

## 3. Technology stack

| Area | Phase 1 choice | Notes |
|---|---|---|
| Base model | `google/gemma-4-E2B-it` | Gemma 4 instruction-tuned multimodal model (E2B) |
| Training env | Google Colab | All model download/load/inference must stay in Colab |
| Inference libs | `transformers>=5.10.1`, `torch`, `accelerate`, `huggingface_hub`, `safetensors` | Verified against current Gemma 4 docs |
| Future training | Transformers + PEFT + TRL (QLoRA/LoRA) | Compatibility must be verified before Phase 2/3 |
| Dataset format | JSONL | Planned for fine-tuning datasets and used for benchmark tasks |
| Model artifacts | Safetensors adapters (future), GGUF (future/Ollama) | No weights committed |
| Runtime | Ollama (future) | Not implemented in Phase 1 |
| Coding agent | Pi (future) | Not implemented in Phase 1 |
| Retrieval | RAG (future) | Not implemented in Phase 1 |

## 4. Repository structure

```text
GemmaCode-E2B/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ VALIDATION_REPORT.md
├─ configs/
│  ├─ model.yaml
│  ├─ training.yaml
│  └─ evaluation.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ train/
│  ├─ validation/
│  └─ test/
├─ notebooks/
│  ├─ 01_baseline.ipynb
│  ├─ 02_dataset_preparation.ipynb
│  ├─ 03_qlora_experiment.ipynb
│  └─ 04_evaluation.ipynb
├─ src/
│  ├─ data/
│  ├─ training/
│  ├─ evaluation/
│  │  ├─ env_diagnostics.py
│  │  ├─ io_utils.py
│  │  └─ rubric.py
│  └─ export/
├─ benchmarks/
│  ├─ coding_tasks.jsonl
│  └─ results/
├─ outputs/
│  ├─ adapters/
│  ├─ checkpoints/
│  └─ merged/
└─ deployment/
   ├─ Modelfile/
   └─ ollama/
```

## 5. Verified Gemma 4 technical requirements (Phase 1)

Primary references (official):
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/docs/transformers/model_doc/gemma4
- https://ai.google.dev/gemma/docs/core/huggingface_inference
- https://ai.google.dev/gemma/docs/capabilities/thinking

Verified/implemented API choices for baseline:
- **Model:** `google/gemma-4-E2B-it`
- **Processor:** `AutoProcessor.from_pretrained(MODEL_ID, token=...)`
- **Model class:** `AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto", low_cpu_mem_usage=True, token=...)`
- **Chat template/generation:** `processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=...)` then `model.generate(...)`
- **Output parsing:** decode generated tokens with `skip_special_tokens=False`, then `processor.parse_response(...)` (with fallback handling)
- **Thinking mode:** supported via `enable_thinking`; E2B behavior differs from larger Gemma 4 models when disabled (template/model-dependent), so the notebook does not hardcode larger-model assumptions
- **Transformers version:** Gemma 4 docs/examples indicate `transformers>=5.10.1`
- **Access:** Gemma models may be gated on Hugging Face; users must accept terms and authenticate with an HF token

> API details can evolve; re-verify against official docs before future phases.

## 6. Google Colab setup (Phase 1)

1. Upload/clone this repository into your Colab environment (or mount from Drive).
2. Use a **GPU runtime** (recommended): `Runtime -> Change runtime type -> GPU`.
3. Open `notebooks/01_baseline.ipynb`.
4. Run cells in order:
   - **A. Environment diagnostics**: prints Python/PyTorch/Transformers versions, CUDA/GPU/VRAM/RAM, and suitability assessment
   - **B. Dependency installation**: installs baseline inference deps in Colab only (`transformers>=5.10.1`, `accelerate`, `huggingface_hub`, `safetensors`, `pyyaml`, `psutil`)
   - **C. Hugging Face authentication**: reads token from Colab Secrets
   - **D. Model loading**: loads `google/gemma-4-E2B-it` in Colab runtime
   - **E. Inference helper**: defines `ask_gemma()` (chat template, deterministic mode, timing/token counts, error handling)
   - **F. Baseline benchmark**: runs 5 tasks from `benchmarks/coding_tasks.jsonl`
   - **G. Save results**: writes local and Google Drive result files

### Important runtime constraints
- Run all model download/load/inference inside **Google Colab** only.
- Do **not** download full model weights to local Windows/Kubuntu machines.
- Do **not** commit model weights, tokens, or private datasets.
- Save outputs to Google Drive for persistence.

## 7. Hugging Face access setup

1. Create/sign in to a Hugging Face account.
2. Accept the license/terms for `google/gemma-4-E2B-it` on its model page (gated access may be required).
3. Create a Hugging Face access token.
4. In Colab: `Secrets` (key icon) -> add `HF_TOKEN` with your token -> grant notebook access.
5. Run the notebook auth cell; it loads `HF_TOKEN` from Colab Secrets and does not hardcode credentials.

If access fails, the notebook reports common causes (terms not accepted, missing/invalid token, network/auth issues).

## 8. How to execute Phase 1

1. Complete Colab and HF setup (sections 6–7).
2. Execute `notebooks/01_baseline.ipynb` top-to-bottom in Colab.
3. Confirm model load succeeds (load time/memory printed).
4. Run the benchmark cell to generate responses for all 5 tasks.
5. Run the save cell to persist results locally and to Google Drive.

### Baseline tasks included
`benchmarks/coding_tasks.jsonl` contains:
1. **Java:** balanced parentheses/brackets/braces
2. **Spring Boot:** diagnose HTTP 403 with JWT authentication
3. **React/TypeScript:** reusable paginated user component
4. **SQL:** highest-paid employee in each department including ties
5. **Software architecture:** music streaming backend (Spring Boot, ASP.NET Core, React, SQL Server)

## 9. How benchmark results are stored

### Local (notebook/runtime workspace)
- Timestamped per-run file: `benchmarks/results/baseline_original_<run_id>.json`
- Convenience latest copy: `benchmarks/results/baseline_original.json` (updated by notebook)
- Run IDs include UTC timestamps to avoid accidental overwrite of prior run files

### Google Drive (Colab)
- Mounted path: `/content/drive`
- Saved to: `MyDrive/GemmaCode-E2B/benchmarks/results/baseline_original_<run_id>.json`

### Result record schema (per task)
Each record includes:
- `task_id`
- `category`
- `prompt`
- `model_identifier`
- `generated_response`
- `generation_duration_seconds`
- `token_counts` (when reliably available)
- `configuration`
- `timestamp`

The notebook also stores run-level metadata (run ID, task file, benchmark config, environment summary, errors if any).

## 10. Evaluation foundation (Phase 1)

`src/evaluation/` provides:
- `env_diagnostics.py`: environment capture and inference suitability assessment
- `rubric.py`: evaluation dimensions and manual scoring template helpers
- `io_utils.py`: JSON/JSONL helpers

Rubric dimensions (for future comparisons: original, fine-tuned, fine-tuned+RAG):
- Technical correctness
- Code quality
- Maintainability
- Instruction following
- Debugging accuracy
- Architectural reasoning
- Hallucination resistance

**No scores are fabricated in Phase 1.** Human evaluation is expected.

## 11. Current limitations

- Phase 1 only (no dataset curation, no fine-tuning, no export, no Ollama/Pi integration)
- No local runtime proof of model load/inference is included (Colab execution required)
- Gemma 4 access is gated and subject to Hugging Face terms/token availability
- Actual performance/VRAM behavior depends on Colab GPU allocation
- Token counts and parsing behavior depend on Transformers/Gemma 4 runtime version; helpers include fallbacks but should be revalidated in Colab

## 12. Upcoming development phases

- **Phase 2 – Dataset preparation:** collect/curate/clean software-engineering data, JSONL normalization, train/val/test splits
- **Phase 3 – QLoRA experiment:** verify Gemma 4 compatibility with Transformers/PEFT/TRL in Colab, run LoRA/QLoRA training, save adapters/checkpoints to Drive
- **Phase 4 – Evaluation:** standardized comparisons (original vs fine-tuned vs fine-tuned+RAG) using rubric and benchmark results

Placeholder notebooks for phases 2–4 are included but not implemented.

## 13. Safety and compliance notes

- Keep all training/inference experiments in Google Colab
- Never download full model weights to local machines
- Never commit model weights, tokens, or private datasets
- Use Colab Secrets for HF authentication
- Avoid unnecessary package upgrades and GPU memory usage
