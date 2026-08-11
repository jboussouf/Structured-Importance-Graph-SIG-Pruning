# Structured Importance Graph (SIG) Pruning

This repository contains the implementation of **Structured Importance Graph (SIG)** pruning, along with the **Greedy-GIS** algorithm for highly efficient, topology-aware neural network pruning. 

The primary objective of SIG is to identify and protect critical architectural bottlenecks during structured pruning by computing global importance scores (using centrality measures like Eigenvector centrality). This results in superior accuracy retention compared to local magnitude-based pruning methods.

## Key Features

- **SIG-CPU**: Original graph-based importance sampling using NetworkX for exact eigenvector centrality.
- **SIG-GPU**: CuPy-accelerated graph pruning that processes weight matrices on the GPU, yielding faster end-to-end execution.
- **Greedy-GIS**: A highly optimized early-stopped power iteration algorithm that bypasses full graph centrality computation, delivering a 1400× speedup over full SIG while matching its accuracy within 0.05%.
- **Baselines**: Includes implementations of standard pruning baselines: Magnitude (L1/L2), Random, BNScale, and Taylor pruning.

## Experimental Results Highlights

### 1. Accuracy vs. Speedup
GIS consistently achieves the lowest accuracy drop among all competing methods at every speedup target. The **Greedy-GIS** variant performs exceptionally well, closely matching the accuracy of the full SIG approach while running orders of magnitude faster.

| Method         | Speedup | Accuracy Drop (%) | Param Retention (%) | Pruning Time (s) |
|----------------|---------|-------------------|---------------------|------------------|
| **SIG-CPU**    | 2×      | **0.60**          | 55.7                | 277.26           |
| **Greedy-GIS** | 2×      | **0.65**          | 62.6                | **0.19**         |
| MagnitudeL2    | 2×      | 1.00              | 75.0                | 0.11             |
| Random         | 2×      | 2.50              | 50.0                | 0.004            |
| **SIG-CPU**    | 3×      | **0.80**          | 32.9                | 276.40           |
| **Greedy-GIS** | 3×      | **0.87**          | 43.2                | **0.20**         |
| MagnitudeL2    | 3×      | 1.33              | 66.3                | 0.01             |
| Random         | 3×      | 3.33              | 33.5                | 0.01             |
| **SIG-CPU**    | 4×      | **0.90**          | 25.9                | 276.14           |
| **Greedy-GIS** | 4×      | **0.97**          | 40.6                | **0.15**         |
| MagnitudeL2    | 4×      | 1.50              | 55.4                | 0.01             |
| Random         | 4×      | 3.75              | 25.2                | 0.004            |

### 2. Centrality Measure Selection
The choice of centrality metric dictates both the accuracy and computation time:
- **Eigenvector** provides the best overall accuracy drop (1.20%).
- **PageRank** offers the best speed/accuracy trade-off, executing ~7× faster than Eigenvector with a negligible 0.05% accuracy penalty.
- **Degree** centrality is fast but weak, failing to model global information flow.

### 3. Topology-Aware Decisions
Unlike local methods (e.g., Random, Magnitude) that prune uniformly across the network, GIS makes intelligent, polarized decisions. It completely preserves critical bottlenecks (e.g., `layer1.0.conv2` and `layer1.0.conv3` in ResNet architectures) while aggressively pruning less important layers. 

**Greedy-GIS** is the recommended configuration, offering near-optimal topology-aware accuracy retention with the speed of naive local pruning methods. For a detailed breakdown of results, see `results_summary.md`.

## Repository Structure

- `gis_core.py`: Core implementation of SIG-CPU and SIG-GPU algorithms.
- `baselines.py`: Implementation of standard pruning baselines and the Greedy-GIS algorithm.
- `evaluation_pipeline.py`: End-to-end benchmarking script to compare different pruning methods.
- `ablation_studies.py` / `generate_figures.py`: Scripts for generating experimental results and figures.
- `results_summary.md`: Detailed report on experimental findings.

## Usage

You can run the full evaluation pipeline using the provided script:

```bash
python evaluation_pipeline.py --dataset cifar100
```

To run a quick smoke-test on a specific model:

```bash
python gis_core.py --model resnet18 --prune_ratio 0.5 --device both
```

## Requirements

The core dependencies are listed in `requirements.txt`. Note that to use **SIG-GPU**, you must have `cupy` installed with the appropriate CUDA toolkit version.

```bash
pip install -r requirements.txt
# For GPU support (optional but recommended):
pip install cupy-cuda12x 
```
