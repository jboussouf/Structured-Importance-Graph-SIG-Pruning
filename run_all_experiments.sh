#!/bin/bash
# run_all_experiments.sh
# Main orchestration script for GIS Benchmarking

echo "=============================================="
echo "Starting GIS Evaluation & Benchmarking Suite"
echo "=============================================="

# 1. Run main evaluations (generates main_evaluation.csv)
echo "[1/3] Running Main Baselines and GIS Evaluation..."
python evaluation_pipeline.py --dataset cifar100

# 2. Run ablation studies (generates ablation_*.csv)
echo "[2/3] Running Ablation Studies (Centrality, Sampling, Layer-wise)..."
python ablation_studies.py

# 3. Generate all figures
echo "[3/3] Generating Publication-Ready Figures..."
python generate_figures.py

echo "=============================================="
echo "All tasks completed! Check the ./results/ and ./figures/ directories."
echo "=============================================="
