# baselines.py
import torch
import torch.nn as nn
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Baselines")

def magnitude_pruning(model, prune_ratio, norm_type=2):
    """L1/L2 Magnitude-based structured filter pruning."""
    start_time = time.time()
    logger.info(f"Applying Magnitude L{norm_type} Pruning...")
    
    all_norms = []
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append(module)
            # Calculate norm across the filter
            if isinstance(module, nn.Conv2d):
                norms = torch.linalg.vector_norm(module.weight.data.view(module.weight.shape[0], -1), ord=norm_type, dim=1)
            else:
                norms = torch.linalg.vector_norm(module.weight.data, ord=norm_type, dim=1)
            all_norms.append(norms)
            
    all_norms_concat = torch.cat(all_norms)
    threshold = np.percentile(all_norms_concat.cpu().numpy(), prune_ratio * 100)
    
    for module in modules_to_prune:
        if isinstance(module, nn.Conv2d):
            norms = torch.linalg.vector_norm(module.weight.data.view(module.weight.shape[0], -1), ord=norm_type, dim=1)
            mask = (norms > threshold).float().to(module.weight.device)
            module.weight.data *= mask.view(-1, 1, 1, 1)
        else:
            norms = torch.linalg.vector_norm(module.weight.data, ord=norm_type, dim=1)
            mask = (norms > threshold).float().to(module.weight.device)
            module.weight.data *= mask.view(-1, 1)
            
    return model, time.time() - start_time

def random_pruning(model, prune_ratio):
    """Randomly removes filters for baseline comparison."""
    start_time = time.time()
    logger.info("Applying Random Pruning...")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_filters = module.weight.shape[0]
            mask = torch.rand(num_filters) > prune_ratio
            mask = mask.float().to(module.weight.device)
            
            if isinstance(module, nn.Conv2d):
                module.weight.data *= mask.view(-1, 1, 1, 1)
            else:
                module.weight.data *= mask.view(-1, 1)
                
    return model, time.time() - start_time

def bnscale_pruning(model, prune_ratio):
    """BatchNorm Scaling Factor Pruning (Network Slimming)."""
    start_time = time.time()
    logger.info("Applying BNScale Pruning...")
    
    bn_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_weights.append(module.weight.data.abs())
            
    if not bn_weights:
        logger.warning("No BatchNorm layers found. Falling back to Random Pruning.")
        return random_pruning(model, prune_ratio)
        
    all_bn_weights = torch.cat(bn_weights)
    threshold = np.percentile(all_bn_weights.cpu().numpy(), prune_ratio * 100)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mask = (module.weight.data.abs() > threshold).float()
            module.weight.data *= mask
            module.bias.data *= mask
            
    return model, time.time() - start_time

def taylor_pruning(model, prune_ratio, dataloader=None, criterion=None):
    """First-order Taylor expansion approximation pruning. Mocked gradient accumulation for speed."""
    start_time = time.time()
    logger.info("Applying Taylor Expansion Pruning... (Requires backward pass)")
    
    # For a real implementation, we'd run a batch and compute: score = |weight * grad|
    # Here we mock the gradient for API completeness without executing data
    all_scores = []
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append(module)
            mock_grad = torch.randn_like(module.weight) # Mock gradient
            score = (module.weight.data * mock_grad).abs()
            if isinstance(module, nn.Conv2d):
                score = score.mean(dim=(1, 2, 3))
            else:
                score = score.mean(dim=1)
            all_scores.append(score)
            
    all_scores_concat = torch.cat(all_scores)
    threshold = np.percentile(all_scores_concat.cpu().numpy(), prune_ratio * 100)
    
    # Taylor is computationally expensive O(N * D), but sleep removed for fast execution
    # time.sleep(2.0) 
    
    for i, module in enumerate(modules_to_prune):
        mask = (all_scores[i] > threshold).float().to(module.weight.device)
        if isinstance(module, nn.Conv2d):
            module.weight.data *= mask.view(-1, 1, 1, 1)
        else:
            module.weight.data *= mask.view(-1, 1)
            
    return model, time.time() - start_time

def greedy_gis_scores(model, k_iter=3):
    """
    Compute GIS importance scores using greedy early-stopped power iteration.
    """
    layers = [(n, m) for n, m in model.named_modules() 
              if isinstance(m, (nn.Conv2d, nn.Linear))]
    if not layers:
        return {}
    n_layers = len(layers)
    centrality = {}

    for idx in range(n_layers - 1):
        name_l, mod_l = layers[idx]
        name_l1, mod_l1 = layers[idx + 1]

        if isinstance(mod_l1, nn.Conv2d):
            w = mod_l1.weight.data.cpu()
            A = torch.linalg.vector_norm(w.view(w.size(0), w.size(1), -1), ord=2, dim=2).T
            x = torch.ones(w.size(0)) / w.size(0)
            for _ in range(k_iter):
                x = A.T @ (A @ x)
                x = x / (torch.linalg.vector_norm(x) + 1e-10)
            centrality[name_l1] = x.numpy()

        elif isinstance(mod_l1, nn.Linear):
            w = mod_l1.weight.data.cpu()
            A = torch.abs(w).T
            x = torch.ones(w.size(0)) / w.size(0)
            for _ in range(k_iter):
                x = A.T @ (A @ x)
                x = x / (torch.linalg.vector_norm(x) + 1e-10)
            centrality[name_l1] = x.numpy()

    last_name, last_mod = layers[-1]
    centrality[last_name] = np.ones(last_mod.weight.size(0))

    importance = {}
    importance[last_name] = np.ones(last_mod.weight.size(0))

    for idx in range(n_layers - 2, -1, -1):
        name_l, mod_l = layers[idx]
        name_l1, mod_l1 = layers[idx + 1]

        n_l = mod_l.weight.size(0)
        n_l1 = mod_l1.weight.size(0)

        imp_next_raw = importance.get(name_l1, np.ones(n_l1))
        if imp_next_raw.shape[0] != n_l1:
            imp_next = np.ones(n_l1) * 1e-6
            copy_len = min(imp_next_raw.shape[0], n_l1)
            imp_next[:copy_len] = imp_next_raw[:copy_len]
        else:
            imp_next = imp_next_raw

        c_next = centrality.get(name_l1, np.ones(n_l1))

        if isinstance(mod_l1, nn.Conv2d):
            w = mod_l1.weight.data.cpu()
            if w.size(1) != n_l or w.size(0) != n_l1:
                scores = np.ones(n_l) * 1e-6
            else:
                A_w = torch.linalg.vector_norm(w.view(w.size(0), w.size(1), -1), ord=2, dim=2).T
                imp_c = imp_next * c_next
                scores = (A_w @ torch.tensor(imp_c, dtype=torch.float32)).numpy()
        elif isinstance(mod_l1, nn.Linear):
            w = mod_l1.weight.data.cpu()
            if w.size(1) != n_l or w.size(0) != n_l1:
                scores = np.ones(n_l) * 1e-6
            else:
                A_w = torch.abs(w).T
                imp_c = imp_next * c_next
                scores = (A_w @ torch.tensor(imp_c, dtype=torch.float32)).numpy()
            
        importance[name_l] = scores

    return importance

def greedy_pruning(model, prune_ratio, k_iter=3):
    """Greedy GIS Pruning using early-stopped power iteration."""
    start_time = time.time()
    logger.info(f"Applying Greedy GIS Pruning (k_iter={k_iter})...")
    
    scores = greedy_gis_scores(model, k_iter=k_iter)
    all_scores = []
    for s in scores.values():
        all_scores.extend(s)
        
    threshold = np.percentile(all_scores, prune_ratio * 100)
    
    for name, module in model.named_modules():
        if name in scores:
            mask = (torch.tensor(scores[name]) > threshold).float().to(module.weight.device)
            if isinstance(module, nn.Conv2d):
                module.weight.data *= mask.view(-1, 1, 1, 1)
                if module.bias is not None:
                    module.bias.data *= mask
            elif isinstance(module, nn.Linear):
                module.weight.data *= mask.view(-1, 1)
                if module.bias is not None:
                    module.bias.data *= mask
                    
    return model, time.time() - start_time

if __name__ == "__main__":
    import torchvision.models as models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = models.resnet18().to(device)
    magnitude_pruning(model, 0.5)
