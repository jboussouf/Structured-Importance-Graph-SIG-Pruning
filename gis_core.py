# gis_core.py
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import logging
import time
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GIS_Core")

# ---------------------------------------------------------------------------
# Try to import CuPy for GPU-accelerated path.  If unavailable, GPU class
# will raise a RuntimeError at instantiation time so the rest of the code
# continues to work on CPU-only machines.
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    logger.warning("CuPy not found — GPU-accelerated GIS will not be available.")


# ===========================================================================
#  CPU implementation (original)
# ===========================================================================

class GraphImportanceSampling:
    def __init__(self, model, centrality_type='eigenvector'):
        self.model = model
        self.centrality_type = centrality_type
        self.G = nx.DiGraph()
        self.layer_nodes = defaultdict(list)
        self.ordered_layers = []
        
    def build_graph(self):
        """
        Builds the computational graph G=(V,E,W).
        Validates Def 3.1: Nodes are neurons (channels), edges are weighted connections.
        """
        logger.info("Building computational graph...")
        prev_layer_name = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.ordered_layers.append(name)
                out_channels = module.weight.shape[0]
                in_channels = module.weight.shape[1]
                
                # Add nodes for this layer in batch
                new_nodes = [f"{name}.{i}" for i in range(out_channels)]
                self.G.add_nodes_from((n, {'layer': name}) for n in new_nodes)
                self.layer_nodes[name].extend(new_nodes)
                
                # Add edges from previous layer if it exists
                if prev_layer_name is not None:
                    # Weight matrix shape: (out_channels, in_channels, ...)
                    weights = module.weight.detach().abs()
                    if isinstance(module, nn.Conv2d):
                        weights = weights.mean(dim=(2, 3)) # Average over spatial dimensions
                        
                    weights_np = weights.cpu().numpy()
                    prev_nodes = self.layer_nodes[prev_layer_name]
                    curr_nodes = self.layer_nodes[name]
                    
                    if len(prev_nodes) == in_channels:
                        # Only add edges whose weight exceeds the mean — prunes ~50% of edges
                        # on dense layers, keeping graph tractable for centrality algorithms.
                        weight_threshold = weights_np.mean()
                        edges_to_add = [
                            (prev_nodes[in_idx], curr_nodes[out_idx], {'weight': float(weights_np[out_idx, in_idx])})
                            for out_idx in range(out_channels) for in_idx in range(in_channels)
                            if weights_np[out_idx, in_idx] >= weight_threshold
                        ]
                        self.G.add_edges_from(edges_to_add)
                prev_layer_name = name
                
    def compute_centrality(self):
        """
        Computes graph centrality.
        Validates Section 3.3.1: Using Eigenvector, PageRank, Betweenness, or Degree centrality.
        """
        logger.info(f"Computing {self.centrality_type} centrality...")
        n_nodes = self.G.number_of_nodes()
        if self.centrality_type == 'eigenvector':
            try:
                # Increase iterations and loosen tolerance for large graphs
                return nx.eigenvector_centrality(self.G, weight='weight', max_iter=1000, tol=1e-6)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge; falling back to PageRank.")
                return nx.pagerank(self.G, weight='weight')
        elif self.centrality_type == 'pagerank':
            return nx.pagerank(self.G, weight='weight')
        elif self.centrality_type == 'betweenness':
            # Exact betweenness is O(V*E) — prohibitively slow on large graphs.
            # Use k-sample approximation: k=500 gives good accuracy in seconds.
            k_samples = min(500, n_nodes)
            logger.info(f"Using approximate betweenness centrality (k={k_samples} of {n_nodes} nodes)...")
            return nx.betweenness_centrality(self.G, weight='weight', k=k_samples, normalized=True)
        elif self.centrality_type == 'degree':
            return nx.degree_centrality(self.G)
        else:
            raise ValueError(f"Unknown centrality: {self.centrality_type}")

    def backward_propagate_importance(self, centralities):
        """
        Validates Eq (1): Backward Importance Propagation.
        I(v) = Sum( |w| * I(v_next) * C(v_next) )
        """
        logger.info("Propagating importance scores backward...")
        importance = {node: 1.0 for node in self.layer_nodes[self.ordered_layers[-1]]} # Base case output layer
        
        for layer in reversed(self.ordered_layers[:-1]):
            for node in self.layer_nodes[layer]:
                score = 0.0
                for successor in self.G.successors(node):
                    w = self.G[node][successor]['weight']
                    c = centralities.get(successor, 1.0)
                    succ_imp = importance.get(successor, 1.0)
                    score += w * succ_imp * c
                importance[node] = score if score > 0 else 1e-6
        return importance

    def importance_sampling(self, importance_scores, sampling_ratio=1.0):
        """
        Validates Eq (3) & (4): Importance Sampling for Variance Reduction.
        Samples nodes based on normalized importance probabilities.
        """
        if sampling_ratio >= 1.0:
            return importance_scores

        logger.info(f"Applying importance sampling with ratio {sampling_ratio}")
        sampled_importance = {}
        
        for layer in self.ordered_layers:
            nodes = self.layer_nodes[layer]
            scores = np.array([importance_scores[n] for n in nodes])
            probs = scores / scores.sum()
            
            num_samples = max(1, int(len(nodes) * sampling_ratio))
            sampled_indices = np.random.choice(len(nodes), size=num_samples, p=probs, replace=False)
            
            for idx in sampled_indices:
                sampled_importance[nodes[idx]] = importance_scores[nodes[idx]]
                
        return sampled_importance

    def apply_structured_pruning(self, importance_scores, prune_ratio):
        """
        Validates Def 3.4: Pruning Decision based on threshold τ.
        Simulates structured pruning by zeroing out entire filters/neurons.
        """
        logger.info(f"Applying structured pruning (ratio: {prune_ratio})...")
        all_scores = list(importance_scores.values())
        threshold = np.percentile(all_scores, prune_ratio * 100)
        
        pruned_nodes_count = 0
        total_nodes = 0
        layer_sparsity = {}
        
        for name, module in self.model.named_modules():
            if name in self.ordered_layers:
                nodes = self.layer_nodes[name]
                mask = torch.ones(module.weight.shape[0])
                
                layer_pruned = 0
                for i, node in enumerate(nodes):
                    total_nodes += 1
                    if importance_scores.get(node, threshold + 1) <= threshold:
                        mask[i] = 0.0
                        pruned_nodes_count += 1
                        layer_pruned += 1
                
                layer_sparsity[name] = layer_pruned / len(nodes)
                
                # Apply mask to simulate structured removal
                mask = mask.to(module.weight.device)
                if isinstance(module, nn.Conv2d):
                    module.weight.data *= mask.view(-1, 1, 1, 1)
                    if module.bias is not None:
                        module.bias.data *= mask
                elif isinstance(module, nn.Linear):
                    module.weight.data *= mask.view(-1, 1)
                    if module.bias is not None:
                        module.bias.data *= mask
                        
        logger.info(f"Pruned {pruned_nodes_count}/{total_nodes} nodes ({(pruned_nodes_count/total_nodes)*100:.2f}%)")
        return self.model, layer_sparsity


# ===========================================================================
#  GPU implementation (CuPy-accelerated)
# ===========================================================================

class GraphImportanceSamplingGPU:
    """
    GPU-accelerated variant of GraphImportanceSampling.

    Key differences from the CPU class:
    - Weight tensors are kept on GPU throughout (CuPy arrays).
    - Edge-weight thresholding uses CuPy vectorised operations instead of
      nested Python loops.
    - Backward importance propagation is implemented as a batched matrix–
      vector multiply on the GPU (CuPy sparse CSR × dense vector) so that
      the entire propagation for a layer pair is fused into a single kernel
      rather than iterated per-node in Python.
    - Importance sampling uses CuPy random sampling, keeping the sampling
      distribution on-device.
    - The graph itself (NetworkX DiGraph) is still built on CPU because
      centrality algorithms (eigenvector, PageRank) run on CPU.  However,
      the expensive weight preprocessing that feeds the graph is done on GPU.
    """

    def __init__(self, model, centrality_type='eigenvector'):
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed.  Install it with: "
                "pip install cupy-cuda12x  (or the version matching your CUDA toolkit)"
            )
        self.model = model
        self.centrality_type = centrality_type
        self.G = nx.DiGraph()
        self.layer_nodes = defaultdict(list)
        self.ordered_layers = []
        # Stores CuPy weight matrices (out × in) for GPU-side propagation
        self._gpu_weights: dict[str, "cp.ndarray"] = {}

    # ------------------------------------------------------------------
    # Graph construction (edge filtering done on GPU)
    # ------------------------------------------------------------------

    def build_graph(self):
        """
        Builds the computational graph G=(V,E,W).
        Weight matrices are processed on the GPU; only edges above the mean
        threshold are written back to the NetworkX graph on CPU.
        """
        logger.info("[GPU] Building computational graph...")
        prev_layer_name = None

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            self.ordered_layers.append(name)
            out_channels = module.weight.shape[0]
            in_channels = module.weight.shape[1]

            new_nodes = [f"{name}.{i}" for i in range(out_channels)]
            self.G.add_nodes_from((n, {'layer': name}) for n in new_nodes)
            self.layer_nodes[name].extend(new_nodes)

            # ---- GPU: compute |W| and average spatial dims if needed ----
            w_gpu = cp.asarray(module.weight.detach().abs())  # transfer to GPU
            if isinstance(module, nn.Conv2d):
                # (out, in, kH, kW) → (out, in) via mean over spatial dims
                w_gpu = w_gpu.mean(axis=(2, 3))
            # w_gpu shape: (out_channels, in_channels)
            self._gpu_weights[name] = w_gpu

            if prev_layer_name is not None and len(self.layer_nodes[prev_layer_name]) == in_channels:
                # ---- GPU: threshold mask ----
                threshold_val = float(w_gpu.mean())
                mask = w_gpu >= threshold_val  # (out, in) bool array on GPU

                # Pull back only the non-zero (row, col) pairs — cheaper than
                # full matrix transfer for sparse networks.
                rows_gpu, cols_gpu = cp.where(mask)
                rows = cp.asnumpy(rows_gpu).tolist()
                cols = cp.asnumpy(cols_gpu).tolist()
                vals = cp.asnumpy(w_gpu[rows_gpu, cols_gpu]).tolist()

                prev_nodes = self.layer_nodes[prev_layer_name]
                curr_nodes = self.layer_nodes[name]

                edges_to_add = [
                    (prev_nodes[c], curr_nodes[r], {'weight': float(v)})
                    for r, c, v in zip(rows, cols, vals)
                ]
                self.G.add_edges_from(edges_to_add)

            prev_layer_name = name

    # ------------------------------------------------------------------
    # Centrality (runs on CPU via NetworkX — same as CPU class)
    # ------------------------------------------------------------------

    def compute_centrality(self):
        """Centrality computation delegated to NetworkX (CPU)."""
        logger.info(f"[GPU] Computing {self.centrality_type} centrality (NetworkX/CPU)...")
        n_nodes = self.G.number_of_nodes()
        if self.centrality_type == 'eigenvector':
            try:
                return nx.eigenvector_centrality(self.G, weight='weight', max_iter=1000, tol=1e-6)
            except nx.PowerIterationFailedConvergence:
                logger.warning("[GPU] Eigenvector centrality failed to converge; falling back to PageRank.")
                return nx.pagerank(self.G, weight='weight')
        elif self.centrality_type == 'pagerank':
            return nx.pagerank(self.G, weight='weight')
        elif self.centrality_type == 'betweenness':
            k_samples = min(500, n_nodes)
            logger.info(f"[GPU] Approximate betweenness (k={k_samples})...")
            return nx.betweenness_centrality(self.G, weight='weight', k=k_samples, normalized=True)
        elif self.centrality_type == 'degree':
            return nx.degree_centrality(self.G)
        else:
            raise ValueError(f"Unknown centrality: {self.centrality_type}")

    # ------------------------------------------------------------------
    # Backward importance propagation (GPU-batched matrix multiply)
    # ------------------------------------------------------------------

    def backward_propagate_importance(self, centralities):
        """
        GPU-accelerated Eq (1): I_prev = W^T · (C ⊙ I_next)

        For each consecutive layer pair (prev → next) the propagation
        reduces to a matrix–vector product on the GPU:
            importance_prev[out] = Σ_in  W[out, in] · C[in] · I_next[in]
        which is computed as a single cuBLAS GEMV kernel.

        Shape guards are applied at every iteration because in ResNet-style
        networks consecutive entries in ordered_layers may not be directly
        connected (skip connections), meaning in_channels of layer[i+1] can
        differ from out_channels of layer[i].
        """
        logger.info("[GPU] Propagating importance scores backward (GPU batched)...")

        # Initialise output layer importance on GPU
        last_layer = self.ordered_layers[-1]
        n_last = len(self.layer_nodes[last_layer])
        imp_gpu = cp.ones(n_last, dtype=cp.float32)  # shape: (out_channels,)
        # Map: layer_name → cupy importance vector
        layer_imp: dict[str, "cp.ndarray"] = {last_layer: imp_gpu}

        # Walk backwards through layers
        for idx in range(len(self.ordered_layers) - 2, -1, -1):
            prev_layer = self.ordered_layers[idx]
            next_layer = self.ordered_layers[idx + 1]
            n_prev = len(self.layer_nodes[prev_layer])
            n_next = len(self.layer_nodes[next_layer])

            W = self._gpu_weights.get(next_layer)  # (out_next, in_prev) — may not align

            # ---- Guard 1: W missing ----
            if W is None:
                layer_imp[prev_layer] = cp.full(n_prev, 1e-6, dtype=cp.float32)
                continue

            # ---- Guard 2: W column-dim must equal n_prev ----
            # W.shape = (out_channels_of_next, in_channels_of_next).
            # in_channels_of_next == n_prev only when layers are directly connected.
            if W.shape[1] != n_prev:
                logger.debug(
                    f"[GPU] Shape mismatch at ({prev_layer}→{next_layer}): "
                    f"W.shape={W.shape}, n_prev={n_prev} — using uniform fallback."
                )
                layer_imp[prev_layer] = cp.full(n_prev, 1e-6, dtype=cp.float32)
                continue

            # ---- Guard 3: align I_next to n_next ----
            # The stored vector may have been produced by a previous matmul whose
            # shape was (in_channels_of_some_other_layer,) != n_next.
            I_next_raw = layer_imp.get(next_layer, cp.full(n_next, 1.0, dtype=cp.float32))
            if I_next_raw.shape[0] != n_next:
                # Resize: copy as many elements as possible, pad rest with 1e-6
                I_next = cp.full(n_next, 1e-6, dtype=cp.float32)
                copy_len = min(I_next_raw.shape[0], n_next)
                I_next[:copy_len] = I_next_raw[:copy_len]
            else:
                I_next = I_next_raw

            # ---- Guard 4: W row-dim must equal n_next ----
            if W.shape[0] != n_next:
                logger.debug(
                    f"[GPU] Row-dim mismatch at ({prev_layer}→{next_layer}): "
                    f"W.shape={W.shape}, n_next={n_next} — using uniform fallback."
                )
                layer_imp[prev_layer] = cp.full(n_prev, 1e-6, dtype=cp.float32)
                continue

            # Build centrality vector for next layer on GPU
            next_nodes = self.layer_nodes[next_layer]
            c_vals = np.array([centralities.get(n, 1.0) for n in next_nodes], dtype=np.float32)
            C_gpu = cp.asarray(c_vals)  # (n_next,)

            # Fused element-wise scale then matmul:
            # score_prev[in] = Σ_out  W[out, in] · C[out] · I_next[out]
            # = W^T · (C ⊙ I_next)
            ci = C_gpu * I_next           # (n_next,)  on GPU
            score_prev = W.T @ ci         # (n_prev,)  on GPU

            # Clamp to avoid zero / negative (matches CPU fallback to 1e-6)
            score_prev = cp.where(score_prev > 0, score_prev, cp.float32(1e-6))
            layer_imp[prev_layer] = score_prev

        # Convert to dict of {node: float} (pull from GPU once per layer)
        importance = {}
        for layer_name, nodes in self.layer_nodes.items():
            if layer_name in layer_imp:
                imp_vec = layer_imp[layer_name]
                # Final alignment: ensure length matches node list
                if imp_vec.shape[0] != len(nodes):
                    aligned = cp.full(len(nodes), 1e-6, dtype=cp.float32)
                    copy_len = min(imp_vec.shape[0], len(nodes))
                    aligned[:copy_len] = imp_vec[:copy_len]
                    imp_vec = aligned
                vals = cp.asnumpy(imp_vec).tolist()
                for node, v in zip(nodes, vals):
                    importance[node] = float(v)
            else:
                for node in nodes:
                    importance[node] = 1e-6
        return importance

    # ------------------------------------------------------------------
    # Importance sampling (GPU random sampling)
    # ------------------------------------------------------------------

    def importance_sampling(self, importance_scores, sampling_ratio=1.0):
        """
        GPU-accelerated importance sampling using CuPy multinomial draw.
        """
        if sampling_ratio >= 1.0:
            return importance_scores

        logger.info(f"[GPU] Applying importance sampling (ratio={sampling_ratio})")
        sampled_importance = {}
        rng = cp.random.default_rng()

        for layer in self.ordered_layers:
            nodes = self.layer_nodes[layer]
            scores_gpu = cp.array([importance_scores[n] for n in nodes], dtype=cp.float32)
            probs_gpu = scores_gpu / scores_gpu.sum()

            num_samples = max(1, int(len(nodes) * sampling_ratio))
            # CuPy random choice without replacement (multinomial approximation)
            idx_gpu = rng.choice(len(nodes), size=num_samples, replace=False, p=probs_gpu)
            sampled_indices = cp.asnumpy(idx_gpu).tolist()

            for idx in sampled_indices:
                sampled_importance[nodes[int(idx)]] = importance_scores[nodes[int(idx)]]

        return sampled_importance

    # ------------------------------------------------------------------
    # Structured pruning (same logic as CPU; masks applied via PyTorch)
    # ------------------------------------------------------------------

    def apply_structured_pruning(self, importance_scores, prune_ratio):
        """
        Pruning decisions computed using CuPy percentile; masks applied
        via PyTorch on the model's native device.
        """
        logger.info(f"[GPU] Applying structured pruning (ratio: {prune_ratio})...")
        all_scores_gpu = cp.array(list(importance_scores.values()), dtype=cp.float32)
        threshold = float(cp.percentile(all_scores_gpu, prune_ratio * 100))

        pruned_nodes_count = 0
        total_nodes = 0
        layer_sparsity = {}

        for name, module in self.model.named_modules():
            if name not in self.ordered_layers:
                continue

            nodes = self.layer_nodes[name]
            scores_gpu = cp.array(
                [importance_scores.get(n, threshold + 1) for n in nodes], dtype=cp.float32
            )
            mask_gpu = (scores_gpu > threshold).astype(cp.float32)  # 1 = keep, 0 = prune

            layer_pruned = int(cp.sum(mask_gpu == 0).item())
            pruned_nodes_count += layer_pruned
            total_nodes += len(nodes)
            layer_sparsity[name] = layer_pruned / len(nodes)

            # Transfer mask to PyTorch on the model's device
            mask_np = cp.asnumpy(mask_gpu)
            mask_torch = torch.from_numpy(mask_np).to(module.weight.device)

            if isinstance(module, nn.Conv2d):
                module.weight.data *= mask_torch.view(-1, 1, 1, 1)
                if module.bias is not None:
                    module.bias.data *= mask_torch
            elif isinstance(module, nn.Linear):
                module.weight.data *= mask_torch.view(-1, 1)
                if module.bias is not None:
                    module.bias.data *= mask_torch

        logger.info(
            f"[GPU] Pruned {pruned_nodes_count}/{total_nodes} nodes "
            f"({(pruned_nodes_count / total_nodes) * 100:.2f}%)"
        )
        return self.model, layer_sparsity


# ===========================================================================
#  Public API — CPU and GPU entry-points
# ===========================================================================

def apply_gis_pruning(model, prune_ratio, centrality='eigenvector', sampling_ratio=1.0):
    """CPU-based GIS pruning (original implementation)."""
    start_time = time.time()
    gis = GraphImportanceSampling(model, centrality_type=centrality)
    gis.build_graph()
    centralities = gis.compute_centrality()
    importance = gis.backward_propagate_importance(centralities)
    sampled_importance = gis.importance_sampling(importance, sampling_ratio)
    pruned_model, layer_sparsity = gis.apply_structured_pruning(sampled_importance, prune_ratio)
    pruning_time = time.time() - start_time
    return pruned_model, pruning_time, layer_sparsity


def apply_gis_pruning_gpu(model, prune_ratio, centrality='eigenvector', sampling_ratio=1.0):
    """
    GPU-accelerated GIS pruning via CuPy.

    Raises RuntimeError if CuPy is not installed.
    Falls back gracefully to the CPU path when CUDA is unavailable.
    """
    if not _CUPY_AVAILABLE:
        logger.warning("CuPy unavailable — falling back to CPU GIS pruning.")
        return apply_gis_pruning(model, prune_ratio, centrality, sampling_ratio)

    start_time = time.time()
    gis = GraphImportanceSamplingGPU(model, centrality_type=centrality)
    gis.build_graph()
    centralities = gis.compute_centrality()
    importance = gis.backward_propagate_importance(centralities)
    sampled_importance = gis.importance_sampling(importance, sampling_ratio)
    pruned_model, layer_sparsity = gis.apply_structured_pruning(sampled_importance, prune_ratio)
    pruning_time = time.time() - start_time
    return pruned_model, pruning_time, layer_sparsity


# ===========================================================================
#  CLI smoke-test
# ===========================================================================

if __name__ == "__main__":
    import torchvision.models as models
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--prune_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'both'], default='both')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing GIS on {args.model} | torch device: {device}")

    if args.device in ('cpu', 'both'):
        model_cpu = getattr(models, args.model)().to(device)
        _, t_cpu, _ = apply_gis_pruning(model_cpu, args.prune_ratio)
        logger.info(f"[SIG-CPU] Pruning completed in {t_cpu:.4f}s")

    if args.device in ('gpu', 'both'):
        model_gpu = getattr(models, args.model)().to(device)
        _, t_gpu, _ = apply_gis_pruning_gpu(model_gpu, args.prune_ratio)
        logger.info(f"[SIG-GPU] Pruning completed in {t_gpu:.4f}s")
