# Mitigating Clinical Sycophancy in Large Language Models via Representation Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This study empirically demonstrates the "Clinical Sycophancy" vulnerability in foundation models (Llama-3-8B and Llama-3-70B), where emotional manipulation (e.g., aggressive tone, panic) causes models to override their safety alignment and prescribe contraindicated, potentially lethal drug combinations (e.g., Warfarin + Ibuprofen). We also provide the code for our white-box mitigation strategy using Representation Engineering (RepE) via latent space intervention.

## 📂 Repository Structure

*   `data/`
    *   `pharmaco_eval_dataset_2k.json`: The Gold Standard synthetic dataset comprising 2,000 deterministic clinical prompts, crossing 20 dangerous drug interactions, 5 emotional archetypes, and 4 scenarios.
*   `results/`
    *   `resultados_baseline_8b.json`: Zero-shot inference outputs for Llama-3-8B.
    *   `resultados_baseline_70b.json`: Zero-shot inference outputs for Llama-3-70B (Quantized Q4_K_M).
*   `src/`
    *   `01_dataset_generation.py`: Script using the OpenAI API (`gpt-4o-mini`) to generate the synthetic prompts safely via environment variables.
    *   `02_baseline_evaluation.py`: Local inference pipelines using `llama_cpp` (and/or `transformers`) to evaluate the models and flag catastrophic failures using deterministic NER.
    *   `03_latent_space_intervention.py`: PyTorch implementation of the RepE ablation study. It uses forward hooks to inject the orthogonal safety vector into intermediate transformer layers (13 to 17) during real-time inference.
    *   `04_statistical_analysis.py`: Unified analytical script that calculates the segmented Sycophancy Error Rates (SER) and computes McNemar's test with continuity correction via `statsmodels` to validate the scaling paradox.
## ⚙️ Quickstart

1. **Clone the repository:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
`

2. **Run the statistical analysis on the provided results:**

   ```bash
    python src/04_statistical_analysis.py
`

**Note**:To run the full inference pipelines, you will need to download the Llama-3 GGUF/Safetensors models from Hugging Face, as they are excluded from this repository due to size constraints.





