# BioLLM Post-Training Lab

**A reproducible research and engineering study of language-model post-training with public biomedical questions, automatically constructed preference data, and verifiable rewards.**

This repository investigates whether fine-tuning changes the quality of biomedical yes/no/maybe answers, agreement with preferred responses, and susceptibility to weak reward functions. The project combines executable Jupyter notebooks, real-public-data connectors, model evaluation, and an independently packaged inference service. It is intended to present the *method*, *evidence*, *code*, and *limitations* together; not to advertise an unverified performance improvement.

**Current evidence:** Notebooks 00–17 have executed locally. The [experiment status](exploration/EXPERIMENT_STATUS.md) summarizes measured results and limitations. The 104-record PubMedQA final test remains sealed, and no model is approved for deployment. Detailed local audit records are excluded from the proposed GitHub publication set.


## 1. What problem does the project solve?

A pretrained **Large Language Model (LLM)** predicts the next token. Instruction tuning can make its output more useful, but biomedical research questions require additional discipline: an answer should be responsive to the supplied abstract, use the correct evidence, avoid unsupported conclusions, and follow a predictable output format. Different post-training methods attempt to improve different aspects of this behavior.

Four hypotheses are examined; none is presumed true:

| Hypothesis | Research question | Comparison |
|---|---|---|
| H1 — Supervision | Does **SFT (Supervised Fine-Tuning)** improve task answers? | Starting model vs. SFT |
| H2 — Preferences | Does **DPO (Direct Preference Optimization)** improve preferred-response behavior? | SFT vs. SFT + DPO |
| H3 — Verifiable rewards | Does **GRPO (Group Relative Policy Optimization)** improve accuracy under a programmatic answer reward? | SFT vs. SFT + GRPO |
| H4 — Reward robustness | Can a model obtain reward by exploiting answer formatting rather than correctness? | Format-only vs. answer-checked rewards |

The completed SFT run uses **LoRA (Low-Rank Adaptation)** with an unquantized base. **QLoRA (Quantized Low-Rank Adaptation)** remains an optional, locally incompatible path. The project also includes a pairwise reward-model experiment, a small **PPO (Proximal Policy Optimization)** optimization example, and a carefully bounded explanation of **RLHF (Reinforcement Learning from Human Feedback)**. The PPO example is *not* full-scale RLHF training on human annotations.

## 2. Data sources: original datasets, fields, and intended use

The data are maintained by their original publishers; downloaded data are cached locally, not committed to this repository. Dataset contents, access methods, and licenses should be reviewed at the source before reuse or redistribution.

| Source | Where to obtain the original data | Content used here | Research use and caveat |
|---|---|---|---|
| **PubMedQA** | [Hugging Face: qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), [original GitHub repository](https://github.com/pubmedqa/pubmedqa), [paper](https://arxiv.org/abs/1909.06146) | Biomedical paper questions, abstract evidence, and `yes` / `no` / `maybe` decisions | SFT, baseline evaluation, and the GRPO answer reward. Use `pqa_labeled` for labeled examples; the `pqa_unlabeled` subset has no reference decision. |
| **Med-REFL-DPO** | [Hugging Face: HANI-LAB/Med-REFL-DPO](https://huggingface.co/datasets/HANI-LAB/Med-REFL-DPO), [original research repository](https://github.com/TianYin123/Med-REFL), [paper](https://arxiv.org/abs/2506.13793) | Prompt/preferred/rejected medical reasoning pairs | DPO and reward-model experiments. These are *automatically constructed* preferences, not independently verified human rankings. |
| **The Cancer Genome Atlas — Lung Adenocarcinoma (TCGA-LUAD)** | [National Cancer Institute Genomic Data Commons project](https://portal.gdc.cancer.gov/projects/TCGA-LUAD), [University of California, Santa Cruz Xena data portal](https://xena.ucsc.edu/download-data/) | Optional public gene-expression, clinical, and survival tables | Domain-specific interpretation extension only. Notebook 14 currently demonstrates an evidence-output contract with explicitly illustrative values; it does **not** execute an analysis of TCGA records. |

### PubMedQA: how records are structured

The dataset's `pqa_labeled` configuration contains approximately **1,000 labeled records** in a Hugging Face `train` split. Key fields are `pubid` (publication identifier), `question`, `context` (a nested record containing `contexts`, a list of abstract passages), `long_answer`, and `final_decision`. `pqa_artificial` and `pqa_unlabeled` are separate configurations and should not be confused with expert-labeled test answers.

Notebook 03 applies an approximately **80% / 10% / 10% publication-aware deterministic split**: a SHA-256 digest of the seed and publication ID (`pubid`) assigns every publication to a stable partition. The notebook persists `exploration/data/pubmedqa_split_manifest.json`, including exact IDs and a source-content fingerprint. All experiments validate against this manifest. This does **not** reproduce PubMedQA's official evaluation split or leaderboard protocol. Published results must identify the exact partition, seed, model checkpoint, and examples used; do not label these estimates "official PubMedQA scores." The PubMedQA project notes an official 500-question test subset available through its source repository.

**Load the original labeled data (download occurs on first execution):**

```python
from datasets import load_dataset

pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(pubmedqa.column_names, len(pubmedqa))
example = pubmedqa[0]
print(example["question"])
print("Abstract:", " ".join(example["context"]["contexts"])[:400])
print("Reference decision:", example["final_decision"])
```

Hugging Face Datasets downloads into its local cache (normally `~/.cache/huggingface/datasets`). For an explicit local snapshot after loading:

```python
pubmedqa.save_to_disk("exploration/data/pubmedqa_labeled")
```

The saved directory is excluded from Git. Check the current dataset revision and record it in an experiment manifest when conducting a controlled research run.

### Med-REFL-DPO: how preference pairs are used

Med-REFL-DPO exposes `reasoning_enhancement` and `reflection_enhancement` configurations. The published authors use a **Tree-of-Thought (ToT)**-based construction process to generate preferred/rejected reasoning trajectories. Notebook 07 inspects the actual record keys before downstream conversion, because datasets may change their schema. A preference example conceptually contains a prompt, a chosen completion, and a rejected completion; validate how the source fields represent those pieces instead of assuming that any generated ranking is expert-verified.

**Retrieve the original preference data:**

```python
from datasets import load_dataset

pairs = load_dataset("HANI-LAB/Med-REFL-DPO", "reasoning_enhancement", split="train")
print(pairs.column_names, len(pairs))
print({key: str(value)[:100] for key, value in pairs[0].items()})
# For local reuse after inspecting the source fields:
pairs.save_to_disk("exploration/data/med_refl_reasoning")
```

DPO and reward-model results from this dataset must be evaluated separately from PubMedQA's yes/no/maybe task. Differences between the two corpora may confound comparisons of training algorithms.

### Optional TCGA-LUAD data

For an additional genomics evidence-analysis study, public Xena-compiled cohort tables may be retrieved through the [Xena data portal](https://xena.ucsc.edu/download-data/). Example published filenames include `TCGA-LUAD.star_counts.tsv.gz`, `TCGA-LUAD.clinical.tsv.gz`, and `TCGA-LUAD.survival.tsv.gz`, hosted by the Xena GDC hub; verify file availability, version, data type, and access constraints in the data portal before use.

**Optional download example (Linux or macOS shell; large files):**

```bash
mkdir -p exploration/data/tcga_luad
curl -fL --retry 3 \
  -o exploration/data/tcga_luad/TCGA-LUAD.star_counts.tsv.gz \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_counts.tsv.gz
curl -fL --retry 3 \
  -o exploration/data/tcga_luad/TCGA-LUAD.clinical.tsv.gz \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.clinical.tsv.gz
curl -fL --retry 3 \
  -o exploration/data/tcga_luad/TCGA-LUAD.survival.tsv.gz \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.survival.tsv.gz
```

Downloading cohort files does not automatically perform expression normalization, differential expression, survival modeling, or clinical validation. Such analyses would require a separately verified, reproducible analysis workflow.

## 3. End-to-end research pipeline

```text
ORIGINAL PUBLIC SOURCES
  PubMedQA (question + nested abstract context + labeled decision)
  Med-REFL-DPO (automatically generated preferred/rejected response pairs)
  TCGA-LUAD (optional public cancer-genomics data)
                  |
                  v
DATA INSPECTION AND PREPARATION — 03, 07
  source revisions -> schema checks -> IDs -> duplicate/group-aware partitions
                  |
                  v
STARTING MODEL AND BASELINE — 04
  Qwen2.5 Instruct -> deterministic outputs -> validation predictions
                  |
                  v
SUPERVISED TRAINING — 05 -> 06
  PubMedQA prompts + decisions -> unquantized LoRA SFT -> adapter + evaluation
                  |
          +-------+-----------+
          |                   |
          v                   v
  PREFERENCE BRANCH     VERIFIABLE REWARD BRANCH
  08 DPO                11 GRPO
  Med-REFL-DPO pairs    PubMedQA decision checking
  SFT -> DPO adapter    SFT -> GRPO adapter
          |                   |
          +--------+----------+
                   |
                   v
MATCHED VALIDATION AND ROBUSTNESS — 12, 17
  identical validation inputs -> measured predictions -> metrics, uncertainty,
  shortcut/reward-hacking checks -> reproducible research summary
                   |
                   v
EXPORT AND RESEARCH RELEASE GATES — 13, 15
  verify merged SFT checkpoint -> registry blocks promotion
                   |
                   v
INDEPENDENT DEPLOYMENT DIRECTORY
  integrated app.py + minimal requirements + Dockerfile + model reference
```

**Side studies:** Notebook 02 explains optimization mathematics; 09 trains an optional scalar reward model from preference pairs; 10 illustrates the clipped PPO surrogate with a synthetic multi-action example; 14 illustrates safe interpretation of biomedical evidence with placeholder values; 16 consolidates technical decisions. The trained branches do not all operate on the same supervision distribution, so the comparison must disclose the training-data differences.

## 4. Repository layout and why the workspaces differ

```text
BioLLM-PostTraining-Lab/
├── README.md                        # This project and original-data guide
├── RELEASE_GUIDE.md                 # Review, packaging and source-control steps
├── exploration/
│   ├── notebooks/                   # 00–17 publication candidates; 18 local-only
│   ├── RESEARCH_PROTOCOL.md         # Experimental protocol and caveats
│   ├── EXPERIMENT_STATUS.md          # Execution and verification status
│   ├── requirements.txt             # Dependencies for experimentation
│   ├── data/                        # Downloaded data; ignored by Git
│   ├── runs/                        # Checkpoints and predictions; ignored by Git
│   └── results/                     # Locally generated measurements; ignored by Git
└── deployment/
    ├── app.py                       # Integrated inference API
    ├── requirements.txt             # Runtime dependencies only
    ├── Dockerfile                    # Service package
    ├── Modelfile.example             # Ollama model configuration example
    └── model/                        # Selected model artifacts; ignored by Git
```

The **exploration** directory is the complete experimental workspace: notebooks explain and execute their procedures directly, from data preparation through analysis. The **deployment** directory is a deliberately smaller service workspace that does not need the experiment history, raw data, or notebooks at runtime. Both directories are in the same repository; build only `deployment/` when creating an inference container.

## 5. How to reproduce the study

### Install and start Jupyter

Use Python 3.10 or a later version compatible with the chosen PyTorch/Transformers/TRL packages. A supported NVIDIA GPU and an appropriately installed **CUDA (Compute Unified Device Architecture)** version are needed for many training variants.

```bash
# From the repository root
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r exploration/requirements.txt
python -m ipykernel install --user --name biollm --display-name "Python (BioLLM)"
python -m notebook
```

Choose the `Python (BioLLM)` kernel. For GPU training, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to install a compatible CUDA build **before** installing the remaining requirements. Libraries evolve; capture package versions, dataset revisions, CUDA version, compute device, seed and model checkpoint for each real experiment.

### Execution controls

CPU examples and repository checks are available without optional downloads or fine-tuning. **Network-dependent and GPU-expensive work must be opted into intentionally.** Set the following in a cell before the corresponding stage (or in the process environment before starting Jupyter):

```python
import os
os.environ["RUN_PUBLIC_DATA"] = "1"  # Inspect original datasets in notebooks 03 / 07
# os.environ["RUN_TRAINING"] = "1"    # Expensive model training in 05 / 08 / 09 / 11
# os.environ["RUN_MODEL_EVAL"] = "1"  # Generate model predictions in 04 / 06 / 12
# os.environ["RUN_TEST_EVAL"] = "1"   # Only after model selection is frozen, notebook 18
# os.environ["EXPERIMENT_FROZEN"] = "1"  # A deliberate final-test authorization
# os.environ["RUN_EXPORT"] = "1"      # Merge trained adapter, notebook 13
# os.environ["RUN_PACKAGE"] = "1"     # Stage selected model files, notebook 15
```

Use `MODEL_ID` to select a consistent model across branches. The examples default to `Qwen/Qwen2.5-0.5B-Instruct` for lower local resource use. Notebook 04 uses the tokenizer's actual Qwen chat template rather than a raw text continuation prompt, and records model identity in an experiment manifest. Scale to `Qwen/Qwen2.5-1.5B-Instruct` only after verifying memory and training compatibility. Training-adapter directories from different base model identifiers are **not interchangeable**.

### Execute in a controlled order

| Notebooks | Scope | Observable output |
|---|---|---|
| 00–03 | Project, GPU, mathematics, data sources and integrity | Hardware summary, illustrative gradients, source schema and split statistics |
| 04 | Starting-model baseline and retrieval proxy | Completed 121-record prediction file: 19 correct; originating-abstract Recall@3 0.9917 |
| 05–06 | SFT and matched validation | Completed unquantized LoRA adapter: 73/121 correct; zero maybe recall |
| 07–09 | Preference data, DPO, reward modeling | DPO adapter completed; revised reward models each scored 9/22 on independent preference test |
| 10–11 | Synthetic PPO and GRPO | CPU bandit demonstration and completed GRPO adapter; training reward is not validation accuracy |
| 12 | Shared validation and paired comparisons | Base/SFT/DPO/GRPO: 19/73/77/76 correct of 121; small DPO/GRPO increments remain uncertain |
| 13–15 | Export, illustrative evidence contract, MLOps | Merged Hugging Face SFT checkpoint; no GGUF/Ollama serving; research registry promotes no model |
| 16–17 | Technical Q&A and reward robustness | Evidence-linked Q&A and six-question base/SFT perturbation probe; no final-test scoring |

**Expected notebook output:** each code cell includes a short explanation of its function and how to interpret printed output. Some outputs are explicit *illustrative checks*, and expensive cells print **skipped** unless activated. A lower training loss or the presence of an adapter file is not proof of an improvement in validation accuracy. A scored prediction file without a compatible `*_manifest.json` is excluded from cross-model comparisons.

## 6. Evaluation and research safeguards

- **Common evaluation conditions:** require identical publication IDs, saved dataset fingerprints, starting-checkpoint identity, role-aware chat prompts, token truncation, deterministic generation, and parsing/scoring logic. Incompatible prediction files are excluded from comparison.
- **Data leakage:** group near-duplicates and related publications before splitting. The SHA-256 publication-ID split prevents exact publication overlap but does not eliminate semantic or cross-dataset near-duplicates. Audit both source-group overlap and semantic leakage before strong scientific claims.
- **Dataset mismatch:** preference experiments use Med-REFL-DPO, whereas yes/no/maybe evaluation uses PubMedQA; report domain differences and do not attribute every shift solely to the optimizer.
- **Measured versus illustrative:** synthetic gradients, toy bandits, reward demonstrations, and fictional hazard ratios verify concepts but do not quantify clinical or model performance.
- **Independent test:** notebook 18 requires explicit opt-in after configuration freeze. Avoid repeated test-set inspection during model selection.
- **Source reproducibility:** record source dataset configuration/revision, prompt template, base checkpoint, random seed, training options, completed steps, and GPU/CPU information.
- **Scientific limitations:** model outputs must not be treated as verified clinical decisions. Observational survival associations do not establish causality or diagnostic validity.

## 7. Deployment boundary

Only after a real checkpoint has been trained and evaluated should an appropriate subset of model artifacts be staged into `deployment/model/`. The deployment package contains its own `app.py`, dependencies, and Dockerfile, so it can be built without the experimental workspace. This does **not** mean that deployment is already live or that an adapter copied into `model/` is automatically usable by the minimal Ollama HTTP backend. Read [deployment/README.md](deployment/README.md) and [RELEASE_GUIDE.md](RELEASE_GUIDE.md) to match the artifact to the serving mechanism.

## 8. Project status and traceability

The pinned starting model is `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The saved PubMedQA `pqa_labeled` split contains 775 training, 121 validation, and 104 sealed final-test publication IDs. Base/SFT/DPO/GRPO answered **19/73/77/76** validation questions correctly; the always-yes reference answered **64**. All trained adapters had zero recall on the 17 `maybe` cases. SFT improved on a weak base, but its nine-case advantage over always-yes and DPO/GRPO's small increments over SFT are uncertain.

The original reward adapter moved from 29/61 to 31/61 on an exposed preference holdout. Two longer development-selected continuations each scored **9/22** on a new group-disjoint preference test, versus 11/22 for their starting adapter. This is not a validated reward signal for Reinforcement Learning from Human Feedback (RLHF) or Proximal Policy Optimization (PPO). Notebook 10's PPO calculation is a synthetic bandit; notebook 14's OncoLncAI example uses invented evidence. Notebook 13 verified a merged float32 Hugging Face SFT checkpoint, but no GGUF/Ollama model or live service. Notebook 15's registry sets `production_model=null` and all entries to `deployment_ready=false`.

The [experiment status](exploration/EXPERIMENT_STATUS.md) records the public evidence boundary. The [publication manifest](PUBLICATION_MANIFEST.md) identifies source files for a future GitHub release. Dataset caches, model weights, local predictions, the private execution audit, and notebook 18 remain local. No deployment or clinical claim follows from these exploratory results.

## 9. Applied Retrieval and Agent Engineering

Model post-training is complemented by **Retrieval-Augmented Generation (RAG)** evaluation, agent orchestration, tool access, and serving design. Notebook 16 contains 25 evidence-linked technical Q&A entries, including these applied engineering topics:

| Theme | Coverage | Practical demonstration |
|---|---|---|
| RAG and Large Language Models (LLMs) | Verify retrieval math with illustrative records; optionally measure TF–IDF matching on original PubMedQA question/abstract pairs; evaluate answers separately | `04_baseline_evaluation.ipynb`, `14_oncolncai_integration_demo.ipynb`, `17_reward_hacking_and_robustness.ipynb` |
| LangChain and LangGraph | Choose chains or state graphs; checkpoint and resume; avoid action loops | `14_oncolncai_integration_demo.ipynb`, `15_mlops_pipeline.ipynb` |
| Agents and Model Context Protocol (MCP) | Select tools; connect APIs through MCP; choose single or multi-agent designs | `14_oncolncai_integration_demo.ipynb` |
| Production and system design | Control token cost and latency; secure tool execution; monitor quality | `15_mlops_pipeline.ipynb`, `17_reward_hacking_and_robustness.ipynb` |

The full Q&A and navigation to each experiment are in **[`16_technical_discussion.ipynb`](exploration/notebooks/16_technical_discussion.ipynb)**. The systems examples include deterministic checks and a bounded actual-model robustness probe. They do not establish a live vector database, LangGraph checkpoint service, or MCP server.

Official architecture references: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture).

## Primary references

- Jin et al., [*PubMedQA: A Dataset for Biomedical Research Question Answering*](https://arxiv.org/abs/1909.06146).
- Yang et al., [*Med-REFL: Medical Reasoning Enhancement via Self-Corrected Fine-grained Reflection*](https://arxiv.org/abs/2506.13793).
- [Hugging Face Transformers Reinforcement Learning (TRL) documentation](https://huggingface.co/docs/trl/main/en/index).
- [Hugging Face Parameter-Efficient Fine-Tuning (PEFT) documentation](https://huggingface.co/docs/peft/index).
- [University of California, Santa Cruz Xena data documentation](https://xena.ucsc.edu/download-data/).


## 10. Evaluation reliability: diagnosing a low score

An earlier 30-question diagnostic from a Qwen2.5-0.5B-Instruct checkpoint produced approximately **0.167 accuracy**, **0.098 macro-F1**, and **0.967 valid-format rate**. These are values observed in a local execution screenshot, **not independently reproduced results in the repository**. Five correct answers among 30 are not sufficient to characterize an LLM's biomedical capabilities or support a post-training improvement claim.

Notebook 04 now directly addresses the main sources of ambiguity:

1. **Role-aware prompting:** the instruction-tuned Qwen tokenizer's chat template is used; manually concatenated `Question/Evidence/Answer` strings are not equivalent to model-native instruction prompts.
2. **Class reference:** actual PubMedQA label frequencies, the majority-class benchmark, and the three-class theoretical uniform-guessing reference are displayed separately.
3. **Exact decoding checks:** deterministic greedy generation and strict one-label parsing; the original response is retained in the CSV for audit.
4. **Failure analysis:** class-specific confusion matrix, malformed answers, context truncation warnings and examples of misclassified papers.
5. **Sample size:** full project validation partition by default; set `EVAL_MAX_ITEMS=30` only to produce a clearly marked diagnostic subset. A subset is not a final benchmark.
6. **Protocol provenance:** the split manifest, model ID, evaluated publication IDs and decoding settings are persisted. Notebook 12 does not mix incompatible experiment files.

**How to interpret the retrieval example:** Query A returns both relevant documents in the top three (Recall@3 = 1.0; Precision@3 = 2/3; Reciprocal Rank = 1/2). Query B returns none (all three scores = 0). The mean Recall@3 is 0.5. These are **illustrative metric-unit tests**, not proof of a functioning or high-quality retrieval system. With `RUN_RETRIEVAL_EVAL=1`, notebook 04 optionally creates a source-backed **TF–IDF (Term Frequency–Inverse Document Frequency) retrieval proxy** from original PubMedQA validation question/abstract pairs. Its gold matching ID is the question's paired abstract; other relevant papers may exist, so it is not a comprehensive human-labeled retrieval benchmark. Neither retrieval experiment measures generated-answer groundedness by itself.

### Windows download and inference warnings

- **Hugging Face cache symlink warning:** Windows may not allow symlinks without Developer Mode or elevated permissions. This generally affects disk consumption/cache duplication, not label accuracy. Developer Mode can enable symlinks; `HF_HUB_DISABLE_SYMLINKS_WARNING=1` hides the message but does not fix the underlying cache behavior.
- **Ignored generation flags:** If a model's configuration contains `temperature`, `top_p`, or `top_k` intended for sampling, Transformers may warn when greedy inference is selected. Notebook 04 constructs an explicit deterministic `GenerationConfig` to avoid inheriting incompatible sampling settings. Record generation parameters when comparing models.
- **PyTorch scaled dot-product attention (SDPA) kernel warnings:** Often indicate attention-kernel selection or fallbacks, not necessarily incorrect model predictions. Confirm PyTorch/CUDA compatibility and GPU memory before inferring a performance problem from a warning alone.

### Executing the revised evaluation

From an activated Python environment with installed research dependencies, start Jupyter in the repository root. Execute notebook 03 with `RUN_PUBLIC_DATA=1` to download PubMedQA and create the split manifest. Then execute notebook 04 with `RUN_MODEL_EVAL=1` to score the model, or `RUN_RETRIEVAL_EVAL=1` for the separate retrieval proxy. Leave `EVAL_MAX_ITEMS=0` for the full project validation partition. After training, notebook 06 evaluates SFT and notebook 12 evaluates all available variants under the same benchmark.

**Current status:** Public-data caches were inspected, local GPU training and validation inference completed, and the retrieval proxy ran. These measured results are summarized in Section 8. The sealed final test, GGUF/Ollama conversion, and production service inference remain unverified.
