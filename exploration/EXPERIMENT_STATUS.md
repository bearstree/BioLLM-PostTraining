# Experiment verification status

Reviewed 2026-10-20. Notebooks 00–17 executed locally. This is a concise publication-status summary, not an independent clinical validation.

| Stage | Verified observation | Evidence boundary |
|---|---|---|
| Data | PubMedQA `pqa_labeled` 1,000 records, split 775 train / 121 validation / 104 sealed final test; Med-REFL-DPO 21,326 pair rows but only 355 exact prompts | Exact-ID/group checks do not eliminate semantic overlap; upstream revisions are unverified. |
| Base | Qwen2.5-0.5B-Instruct, pinned revision; 19/121 validation correct, macro-F1 0.115 | Mostly predicts maybe; below majority-yes 64/121. |
| SFT | 40-step unquantized LoRA run on 200 train IDs; 73/121 validation correct, macro-F1 0.422 | Nine above majority-yes is uncertain; zero maybe recall. Four-bit QLoRA did not run. |
| DPO | SFT-derived, 120 distinct-prompt pairs, 30 steps; 77/121 PubMedQA validation correct | Four-case gain over SFT gives exact paired p = 0.125; no independent Med-REFL preference gain established. |
| Reward model | Original exposed holdout 29/61 to 31/61; two longer seeds each 9/22 on new group-disjoint preference test | No validated RLHF/PPO reward signal. |
| PPO | Synthetic three-action CPU bandit expected reward 0.4000 to 0.9963 | No biomedical language-model PPO run. |
| GRPO | 40 rollout steps; 76/121 PubMedQA validation correct, macro-F1 0.408 | +3 net cases versus SFT, paired p about 0.664; zero maybe recall. |
| Robustness | All variants format-valid on 121 validation rows despite 19–77 correct; a six-case base/SFT prompt probe recorded 36 generations | Small perturbation diagnostic, not a general attack-success estimate. |
| Export and MLOps | Merged float32 Hugging Face SFT checkpoint reloaded; registry `production_model=null` | No GGUF/Ollama model, service smoke, deployment promotion, or live monitoring. |

The PubMedQA final 104 records remain unscored. Validation has been inspected repeatedly; freeze a candidate and analysis plan independently before a one-time final evaluation. The local integrated review and execution audit retain detailed private history and are excluded from the proposed GitHub publication set.
