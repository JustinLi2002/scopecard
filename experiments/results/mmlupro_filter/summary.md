# MMLU-Pro filter study: results

Design: `experiments/MMLUPRO_FILTER_DESIGN.md` (fixed 2026-11-15). Outputs from TIGER-AI-Lab/MMLU-Pro at commit `f418b116db`; 46 systems kept. M = MMLU-origin questions (filtered by the eight models), O = questions from other sources. Accuracies are standardised over 8 categories that have at least 100 questions from each source. 1 point = 0.01.

Corrections of 2026-11-16 (design, Section 9). (1) Excluded from every analysis because the question ids do not correspond to the current questions: Meta-Llama-3-70B-Instruct_5shots (10756 of 11962 recorded gold answers differ from the table). (2) The reference tail share is the share of subsets of the placebo pool, of the size of the filter set, whose mean is at least the observed value. It says where that value lies among those subsets. It is not a calibrated p-value: the filter models are not exchangeable with the pool systems. The readings below are the labels that the rule fixed in the design gives.

## Primary analysis

### Standardised over eligible categories, siblings among the controls
- Filter models: a(O) from 21.7% to 32.8%; controls per filter model: 15, 6, 4, 11, 8, 4; placebo pool: 15 non-filter systems.
- **Penalty P = +1.57 points** (question bootstrap 95% interval +0.63 to +2.60); placebo penalties of the pool: mean +0.16, SD 3.97, range -9.18 to +7.01; **reference tail share: 687 of 5005 subsets of the pool (0.1373; exact)**.
- By filter model: gemma-7b_5shots +0.85 [-1.19, +2.90]; Llama-2-13b-hf_5shots +4.13 [+2.07, +6.03]; Llama-2-7b-hf_5shots +7.23 [+5.24, +9.10]; Mistral-7B-v0.1_5shots +1.50 [-0.36, +3.22]; Yi-6b-Chat_5shots -1.99 [-4.02, -0.28]; Yi-6B_5shots -2.28 [-4.34, -0.23].
- Siblings (placebo penalties): Llama-2-70b-hf_5shots -0.22; Mistral-7B-Instruct-v0.1_5shots +7.01; Mistral-7B-Instruct-v0.2_5shots -0.55; Mistral-7B-v0.2-hf_5shots +3.43; Yi-34B_5shots -1.47.
- Pool systems with a placebo penalty at least as large as P: 4 of 15.

**Reading fixed in the design: not detected.**

## Secondary analyses

### 1. Siblings removed from the controls and from the placebo pool
- Filter models: a(O) from 21.7% to 32.8%; controls per filter model: 10, 3, 3, 6, 4, 3; placebo pool: 10 non-filter systems.
- **Penalty P = +3.37 points** (question bootstrap 95% interval +2.08 to +4.72); placebo penalties of the pool: mean +0.00, SD 4.27, range -9.23 to +5.53; **reference tail share: 0 of 210 subsets of the pool (0.0000; exact)**.
- By filter model: gemma-7b_5shots +1.92 [-0.19, +4.01]; Llama-2-13b-hf_5shots +5.81 [+3.58, +8.10]; Llama-2-7b-hf_5shots +9.95 [+7.88, +12.06]; Mistral-7B-v0.1_5shots +2.37 [+0.32, +4.27]; Yi-6b-Chat_5shots -0.28 [-2.34, +1.63]; Yi-6B_5shots +0.44 [-1.79, +2.84].
- Pool systems with a placebo penalty at least as large as P: 2 of 10.

### 2. Regression version
- a(M) regressed on a(O) and its square over the 40 non-filter systems. Mean shortfall of the filter models below the fitted curve: +1.13 points (by model: gemma-7b_5shots +0.11, Llama-2-13b-hf_5shots +3.76, Llama-2-7b-hf_5shots +6.81, Mistral-7B-v0.1_5shots +1.10, Yi-6b-Chat_5shots -3.20, Yi-6B_5shots -1.82); leave-one-out shortfalls of the pool: mean -0.27, SD 3.89; reference tail share: 677 of 5005 subsets of the pool (0.1353; exact).

### 3. Dependence footprint
- Mean pairwise phi among the filter models: on M 0.303, on O 0.381; difference -0.078. The same difference for 2000 random sets of 6 pool systems: mean -0.066, central 95% -0.094 to -0.037; share at or below the filter models' value: 0.2150.

### 4. All categories, no standardisation
- Filter models: a(O) from 19.0% to 29.2%; controls per filter model: 15, 7, 4, 12, 7, 6; placebo pool: 15 non-filter systems.
- **Penalty P = +1.56 points**; placebo penalties of the pool: mean -0.14, SD 3.61, range -8.24 to +5.19; **reference tail share: 343 of 5005 subsets of the pool (0.0685; exact)**.
- By filter model: gemma-7b_5shots +1.43; Llama-2-13b-hf_5shots +3.62; Llama-2-7b-hf_5shots +5.54; Mistral-7B-v0.1_5shots +1.22; Yi-6b-Chat_5shots -2.08; Yi-6B_5shots -0.37.
- Siblings (placebo penalties): Llama-2-70b-hf_5shots -2.25; Mistral-7B-Instruct-v0.1_5shots +5.19; Mistral-7B-Instruct-v0.2_5shots -1.69; Mistral-7B-v0.2-hf_5shots +2.49; Yi-34B_5shots -3.49.
- Pool systems with a placebo penalty at least as large as P: 6 of 15.

## Systems in the ability range (standardised accuracies, in %)

| System | Role | a(O) | a(M) | a(M) minus a(O) |
|---|---|---:|---:|---:|
| Llama-2-7b-hf_5shots | filter | 21.7 | 16.3 | -5.4 |
| Yi-6B_5shots | filter | 24.2 | 28.3 | +4.1 |
| Llama-2-13b-hf_5shots | filter | 24.8 | 23.5 | -1.2 |
| Yi-6b-Chat_5shots | filter | 25.8 | 31.8 | +6.0 |
| Mistral-7B-Instruct-v0.1_5shots | sibling | 27.0 | 24.9 | -2.1 |
| Qwen1.5-7B-Chat_5shots | other | 27.8 | 31.6 | +3.8 |
| Mistral-7B-Instruct-v0.2_5shots | sibling | 28.7 | 33.0 | +4.3 |
| Mistral-7B-v0.2-hf_5shots | sibling | 29.5 | 30.9 | +1.3 |
| Mistral-7B-v0.1_5shots | filter | 29.8 | 32.8 | +3.0 |
| gemma-7b_5shots | filter | 32.8 | 37.5 | +4.7 |
| Meta-Llama-3-8B_5shots | other | 34.4 | 39.4 | +5.0 |
| c4ai-command-r-v01_5shots | other | 34.6 | 39.6 | +4.9 |
| Llama-2-70b-hf_5shots | sibling | 35.2 | 41.0 | +5.8 |
| Qwen1.5-14B-Chat_5shots | other | 35.5 | 44.9 | +9.3 |
| Meta-Llama-3_1-8B_5shots | other | 37.4 | 39.6 | +2.2 |
| Mixtral-8x7B-v0.1_5shots | other | 39.3 | 46.0 | +6.7 |
| Yi-34B_5shots | sibling | 39.4 | 47.1 | +7.7 |
| Phi-3-mini-4k-instruct_5shots | other | 41.0 | 50.8 | +9.8 |
| Mixtral-8x7B-Instruct-v0.1_5shots | other | 41.1 | 48.2 | +7.1 |
| Meta-Llama-3-8B-Instruct_5shots | other | 41.8 | 43.6 | +1.9 |
| Qwen1.5-72B-Chat_5shots | other | 42.0 | 57.6 | +15.6 |

