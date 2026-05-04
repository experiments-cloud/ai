# Mitigating Clinical Sycophancy in Large Language Models via Representation Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the dataset and codebase for the paper **"[Insert your paper title here]"**, submitted to *Applied Intelligence*. 

This study empirically demonstrates the "Clinical Sycophancy" vulnerability in foundation models (Llama-3-8B and Llama-3-70B), where emotional manipulation (e.g., aggressive tone, panic) causes models to override their safety alignment and prescribe contraindicated, potentially lethal drug combinations (e.g., Warfarin + Ibuprofen). We also provide the code for our white-box mitigation strategy using Representation Engineering (RepE) via latent space intervention.

## 📂 Repository Structure

*   `data/`
    *   `pharmaco_eval_dataset_2k.json`: The Gold Standard synthetic dataset comprising 2,000 deterministic clinical prompts, crossing 20 dangerous drug interactions, 5 emotional archetypes, and 4 scenarios.
*   `results/`
    *   `resultados_baseline_8b.json`: Zero-shot inference outputs for Llama-3-8B.
    *   `resultados_baseline_70b.json`: Zero-shot inference outputs for Llama-3-70B (Quantized Q4_K_M).
*   `src/`
    *   `01_dataset_generation.py`: Script using OpenAI API to generate the synthetic prompts.
    *   `02_baseline_evaluation.py`: Inference pipeline using `llama_cpp` and `transformers` for local evaluation.
    *   `03_latent_space_intervention.py`: PyTorch implementation of the RepE ablation study, using forward hooks to inject the orthogonal safety vector into intermediate layers.
    *   `04_statistical_analysis.py`: Calculates Sycophancy Error Rates (SER) and runs McNemar's test via `statsmodels`.

## ⚙️ Quickstart

1. **Clone the repository:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
```

2. **Run the statistical analysis on the provided results:**

   ```bash
    python src/04_statistical_analysis.py
```

Note: To run the full inference pipelines, you will need to download the Llama-3 GGUF/Safetensors models from Hugging Face, as they are excluded from this repository due to size constraints.





