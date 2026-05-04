"""
04_statistical_analysis.py

This script performs the comprehensive statistical analysis for the paper.
It reads the inference results from both Llama-3-8B and Llama-3-70B models and:

1. Calculates the Sycophancy Error Rate (SER) segmented by emotional tone, 
   clinical scenario, and base drug for the 70B model.
2. Constructs a 2x2 contingency table matching the paired responses of both models.
3. Computes McNemar's test to statistically validate the "scaling paradox" 
   (whether the higher capacity model fails significantly more under emotional pressure).

Dependencies:
    pip install statsmodels
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any
from statsmodels.stats.contingency_tables import mcnemar

# --- Configuration ---
RESULTS_8B_PATH = "results/results_baseline_8b.json"
RESULTS_70B_PATH = "results/results_baseline_70b.json"

def load_json(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSON file safely."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}. Please run inference scripts first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def print_breakdown_stats(title: str, stats_dict: Dict[str, Dict[str, int]]) -> None:
    """Helper function to print beautifully formatted breakdown tables."""
    print(f"\n--- {title} ---")
    # Sort descending by Sycophancy Error Rate (SER)
    sorted_stats = sorted(stats_dict.items(), key=lambda x: (x[1]["fails"] / x[1]["total"]), reverse=True)
    
    for key, val in sorted_stats:
        rate = (val["fails"] / val["total"]) * 100
        print(f"{key[:40]:<40} | Fails: {val['fails']:>4}/{val['total']:>4} | SER: {rate:.2f}%")

def analyze_70b_vulnerabilities(data_70b: List[Dict[str, Any]]) -> None:
    """Analyzes and prints the error distribution for the 70B model."""
    total_cases = len(data_70b)
    total_fails = sum(1 for d in data_70b if d.get("failed_baseline", 0) == 1)

    fails_by_tone = defaultdict(lambda: {"total": 0, "fails": 0})
    fails_by_drug = defaultdict(lambda: {"total": 0, "fails": 0})
    fails_by_scenario = defaultdict(lambda: {"total": 0, "fails": 0})

    for item in data_70b:
        tone = item.get("tone", "Unknown")
        drug = item.get("base_drug", "Unknown")
        scenario = item.get("scenario", "Unknown")
        is_fail = item.get("failed_baseline", 0)
        
        fails_by_tone[tone]["total"] += 1
        fails_by_tone[tone]["fails"] += is_fail
        
        fails_by_drug[drug]["total"] += 1
        fails_by_drug[drug]["fails"] += is_fail
        
        fails_by_scenario[scenario]["total"] += 1
        fails_by_scenario[scenario]["fails"] += is_fail

    print("\n" + "="*65)
    print("=== PART 1: LLAMA-3-70B SYCOPHANCY DISTRIBUTION ===")
    print(f"Total Evaluated: {total_cases} | Global Error Rate: {(total_fails/total_cases)*100:.2f}%")
    print("="*65)

    print_breakdown_stats("VULNERABILITY BY EMOTIONAL ARCHETYPE", fails_by_tone)
    print_breakdown_stats("VULNERABILITY BY CLINICAL SCENARIO", fails_by_scenario)
    print_breakdown_stats("VULNERABILITY BY BASE DRUG", fails_by_drug)

def run_mcnemars_test(data_8b: List[Dict[str, Any]], data_70b: List[Dict[str, Any]]) -> None:
    """Runs McNemar's test to compare the performance of 8B vs 70B models."""
    print("\n" + "="*65)
    print("=== PART 2: ARCHITECTURAL SCALE COMPARISON (MCNEMAR'S TEST) ===")
    print("="*65)

    # Index 70B results using the exact prompt to ensure perfect pairing
    data_70b_dict = {item["user_prompt"]: item.get("failed_baseline", 0) for item in data_70b}

    failed_both = 0
    passed_both = 0
    failed_70b_passed_8b = 0
    failed_8b_passed_70b = 0

    for item in data_8b:
        prompt = item["user_prompt"]
        fail_8b = item.get("failed_baseline_8b", 0)
        
        if prompt in data_70b_dict:
            fail_70b = data_70b_dict[prompt]
            
            if fail_70b == 1 and fail_8b == 1:
                failed_both += 1
            elif fail_70b == 0 and fail_8b == 0:
                passed_both += 1
            elif fail_70b == 1 and fail_8b == 0:
                failed_70b_passed_8b += 1
            elif fail_70b == 0 and fail_8b == 1:
                failed_8b_passed_70b += 1

    # 2x2 Contingency Table for McNemar
    contingency_table = [
        [failed_both, failed_70b_passed_8b],
        [failed_8b_passed_70b, passed_both]
    ]

    # Calculate McNemar's with continuity correction
    result = mcnemar(contingency_table, exact=False, correction=True)

    print("\n--- 2x2 CONTINGENCY MATRIX (8B vs 70B) ---")
    print(f"Both Models Failed (Sycophancy):      {failed_both}")
    print(f"Both Models Passed (Safe):            {passed_both}")
    print(f"70B Failed BUT 8B Passed (Discordant):{failed_70b_passed_8b}")
    print(f"8B Failed BUT 70B Passed (Discordant):{failed_8b_passed_70b}")

    print("\n--- STATISTICAL RESULTS ---")
    print(f"Chi-squared Statistic: {result.statistic:.4f}")
    print(f"p-value:               {result.pvalue:.4e}")

    if result.pvalue < 0.05:
        print("Conclusion: The difference in safety performance is STATISTICALLY SIGNIFICANT (p < 0.05).")
        print("This empirically validates the Scaling Paradox for Clinical Sycophancy.")
    else:
        print("Conclusion: The difference is NOT statistically significant.")

def main():
    print("Loading inference logs...")
    data_8b = load_json(RESULTS_8B_PATH)
    data_70b = load_json(RESULTS_70B_PATH)
    
    # 1. Segmented Analysis
    analyze_70b_vulnerabilities(data_70b)
    
    # 2. Comparative Analysis
    run_mcnemars_test(data_8b, data_70b)

if __name__ == "__main__":
    main()