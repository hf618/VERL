import torch
import torch.nn.functional as F
import time

def compute_single_entropy(hidden: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'gram') -> float:
    """
    Automatically chooses the smallest matrix (L x L or D x D) to compute spectral entropy.
    Mathematically equivalent to the original version regardless of matrix_type.
    """
    # matrix_type argument is kept for API compatibility but ignored for logic optimization
    if hidden.size(0) < 2: return 0.0
    try:
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)
        
        L, D = centered.shape
        
        # --- Adaptive Optimization ---
        # Always compute the smaller matrix to save time/memory
        if L < D:
            # Short sequence: L x L
            G = centered @ centered.T
        else:
            # Long sequence: D x D
            G = centered.T @ centered
        
        # Note: We do NOT need to divide by (L-1) even for 'covariance' intent,
        # because entropy is scale-invariant (normalization cancels the scale).
        
        eigvals = torch.linalg.eigvalsh(G) 
        
        # Safety Truncation: remove theoretical noise zeros
        k = min(L, D)
        eigvals = eigvals[-k:]
        
        # Filter numerical zeros
        eigvals = eigvals[eigvals > 1e-8]
        if len(eigvals) == 0: return 0.0
        
        normalized = eigvals / eigvals.sum()
        if abs(alpha - 1.0) < 1e-6:
            normalized = normalized[normalized > 1e-12]
            return -torch.sum(normalized * torch.log(normalized)).item()
        else:
            return (1/(1-alpha)) * torch.log(torch.sum(normalized**alpha)).item()
    except torch._C._LinAlgError:
        return 0.0
    
def compute_single_effective_rank(hidden: torch.Tensor, svd_rank: int, svd_niter: int, log_output: bool = False, method: str = 'lowrank') -> tuple[float, float]:
    """
    [Adaptive Optimized Version with Safety Truncation]
    """
    if hidden.size(0) < 2: return 0.0, 0.0
    try:
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)

        L, D = centered.shape
        # Adaptive Choice
        if L < D:
            G = centered @ centered.T
        else:
            G = centered.T @ centered
        
        eigvals = torch.linalg.eigvalsh(G)
        
        # --- Safety Truncation ---
        # Ensure we strictly use at most min(L, D) eigenvalues to match SVD behavior perfectly
        # Even if Adaptive algo already produces correct size, this is a zero-cost safety net.
        k = min(L, D)
        eigvals = eigvals[-k:] 
        
        S = torch.sqrt(torch.relu(eigvals))
            
        traditional_rank = 0
        if S is not None and S.numel() > 0:
            tol = S.max() * max(centered.shape) * torch.finfo(S.dtype).eps
            traditional_rank = torch.sum(S > tol).item()
        else:
            return 0.0, 0.0

        normalized_S = S / (S.sum() + 1e-8)
        effective_rank_val = 0.0
        if log_output:
            effective_rank_val = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
        else:
            effective_rank_val = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8))).item()
            
        return effective_rank_val, float(traditional_rank)
    except torch._C._LinAlgError:
        return 0.0, 0.0
    

def compute_single_curvature(hidden: torch.Tensor) -> float:
    """Calculate the curvature of a single sample"""
    if hidden.size(0) < 3: return 0.0
    diffs = hidden[1:] - hidden[:-1]
    angles = []
    chunk_size = 256
    for chunk in torch.split(diffs, chunk_size, dim=0):
        if chunk.size(0) < 2: continue
        norms = torch.norm(chunk, dim=1, keepdim=True)
        valid = (norms > 1e-6).squeeze()
        chunk = chunk[valid]
        if chunk.size(0) < 2: continue
        cos_sim = F.cosine_similarity(chunk[:-1], chunk[1:], dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angles.append(torch.arccos(cos_sim))
    if angles:
        return torch.cat(angles).mean().item()
    return 0.0

def _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names):
    """Auxiliary function: Calculate all required metrics based on the eigenvalues of the Gram matrix."""
    S = torch.sqrt(torch.relu(eigenvalues))
    results = []
    for name in selected_metric_names:
        if name == "Response Entropy 1":
            eigvals = eigenvalues[eigenvalues > 1e-8]
            if len(eigvals) == 0:
                results.append(0.0)
                continue
            normalized = eigvals / eigvals.sum()
            normalized = normalized[normalized > 1e-12]
            entropy = -torch.sum(normalized * torch.log(normalized)).item()
            results.append(entropy)
        elif name in ["Effective Rank", "Log Effective Rank", "Traditional Rank"]:
            normalized_S = S / (S.sum() + 1e-8)
            shannon_entropy_S = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
            if name == "Effective Rank":
                results.append(torch.exp(torch.tensor(shannon_entropy_S)).item())
            elif name == "Log Effective Rank":
                results.append(shannon_entropy_S)
            elif name == "Traditional Rank":
                if S.numel() > 0:
                    tol = S.max() * max(S.shape) * torch.finfo(S.dtype).eps
                    trad_rank = torch.sum(S > tol).item()
                    results.append(float(trad_rank))
                else:
                    results.append(0.0)
        elif name == "Curvature":
            results.append(0.0)
        else:
            results.append(0.0)
    return results

def calculate_diffs_for_single_sample_optimized(valid_hidden, max_seq_len, stride, selected_metric_names, 
                                                svd_rank, svd_niter, svd_method):
    """
    [Final Adaptive Production Version]
    """
    valid_len = valid_hidden.size(0)
    hidden_dim = valid_hidden.size(1)
    
    if valid_len > max_seq_len:
        valid_hidden = valid_hidden[-max_seq_len:]
        valid_len = max_seq_len

    per_stride_diffs_i = {f"{name} diff": [] for name in selected_metric_names}
    per_stride_diffs_i.update({f"{name} diff 2": [] for name in selected_metric_names})
    per_stride_diffs_i.update({f"{name} diff_timing": 0.0 for name in selected_metric_names})
    
    if valid_len < 2 * stride:
        return per_stride_diffs_i

    shared_metrics = [n for n in selected_metric_names if n != "Curvature"]
    primary_payer_metric = shared_metrics[0] if len(shared_metrics) > 0 else None
    has_curvature = "Curvature" in selected_metric_names

    history_sum = [0.0] * len(selected_metric_names)
    history_count = 0
    prev_diff = None

    use_d_mode = valid_len >= hidden_dim

    s = torch.zeros(1, hidden_dim, device=valid_hidden.device, dtype=torch.float32)
    processed_idx = 0
    U = None      
    G_raw = None  

    for t in range(stride, valid_len, stride):
        current_window = valid_hidden[:t+1]
        
        # A. Shared Computation
        t0_shared = time.perf_counter()
        current_window_f32 = current_window.to(torch.float32)
        
        new_chunk = current_window_f32[processed_idx:]
        s = s + new_chunk.sum(dim=0, keepdim=True)
        k = current_window_f32.shape[0]

        if not use_d_mode:
            # L-Mode
            if processed_idx > 0:
                H_old = current_window_f32[:processed_idx]
                C12 = H_old @ new_chunk.T
                C22 = new_chunk @ new_chunk.T
                top_part = torch.cat([U, C12], dim=1)
                bottom_part = torch.cat([C12.T, C22], dim=1)
                U = torch.cat([top_part, bottom_part], dim=0)
            else:
                U = new_chunk @ new_chunk.T
            
            mean_vec = s / k
            mean_gram = mean_vec @ mean_vec.T
            hs_T = current_window_f32 @ s.T / k
            ones_k = torch.ones((k, 1), device=current_window_f32.device, dtype=torch.float32)
            G = U - hs_T @ ones_k.T - ones_k @ hs_T.T + mean_gram
            
        else:
            # D-Mode
            term = new_chunk.T @ new_chunk
            if G_raw is None:
                G_raw = term
            else:
                G_raw = G_raw + term
            G = G_raw - (s.T @ s) / k
        
        processed_idx += new_chunk.shape[0]

        eigenvalues = torch.linalg.eigvalsh(G)
        
        # --- Safety Truncation ---
        # Only keep the top min(k, hidden_dim) eigenvalues.
        # This filters out numerical noise if the matrix dim > rank.
        valid_rank_limit = min(k, hidden_dim)
        eigenvalues = eigenvalues[-valid_rank_limit:]
        
        current_metrics = _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names)
        
        t1_shared = time.perf_counter()
        shared_cost = t1_shared - t0_shared 

        # B. Independent Computation
        curvature_cost = 0.0
        if has_curvature:
            t0_curv = time.perf_counter()
            current_metrics[selected_metric_names.index("Curvature")] = compute_single_curvature(current_window)
            curvature_cost = time.perf_counter() - t0_curv

        # C. Diff Storage
        if history_count > 0:
            hist_avg = [sm / history_count for sm in history_sum]
            curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
            
            for idx, name in enumerate(selected_metric_names): 
                per_stride_diffs_i[f"{name} diff"].append(curr_diff[idx])
                
                if name == "Curvature":
                    per_stride_diffs_i[f"{name} diff_timing"] += curvature_cost
                elif name == primary_payer_metric:
                    per_stride_diffs_i[f"{name} diff_timing"] += shared_cost

            if prev_diff is not None:
                curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                for idx, name in enumerate(selected_metric_names): 
                    per_stride_diffs_i[f"{name} diff 2"].append(curr_diff2[idx])
                    
            prev_diff = curr_diff
            
        history_sum = [sm + curr for sm, curr in zip(history_sum, current_metrics)]
        history_count += 1
            
    return per_stride_diffs_i

# We dont use this
def calculate_diffs_for_single_sample_original(valid_hidden, max_seq_len, stride, selected_metric_names, 
                                      svd_rank, svd_niter, svd_method):
    """Compute the first and second order differences of all selected metrics for the hidden state of a single sample. (Baseline version)"""
    metric_calculators = {
        "Response Entropy 1": lambda h: compute_single_entropy(h, 1.0001, "gram"),
        "Curvature": lambda h: compute_single_curvature(h),
        "Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=False, method=svd_method)[0],
        "Log Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=True, method=svd_method)[0],
        "Traditional Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, method=svd_method)[1]
    }
    active_calculators = [metric_calculators[name] for name in selected_metric_names if name in metric_calculators]
    num_metrics_to_track = len(active_calculators)
    valid_len = valid_hidden.size(0)
    if valid_len > max_seq_len:
        valid_hidden = valid_hidden[-max_seq_len:]
        valid_len = max_seq_len

    history_sum, history_count, prev_diff = [0.0] * num_metrics_to_track, 0, None
    per_stride_diffs_i = {f"{name} diff": [] for name in selected_metric_names}
    per_stride_diffs_i.update({f"{name} diff 2": [] for name in selected_metric_names})

    for t in range(stride, valid_len, stride):
        sub_hidden = valid_hidden[:t+1]
        current_metrics = [calc(sub_hidden) for calc in active_calculators]
        if history_count > 0:
            hist_avg = [s / history_count for s in history_sum]
            curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
            for idx, name in enumerate(selected_metric_names): 
                per_stride_diffs_i[f"{name} diff"].append(curr_diff[idx])
            if prev_diff is not None:
                curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                for idx, name in enumerate(selected_metric_names): 
                    per_stride_diffs_i[f"{name} diff 2"].append(curr_diff2[idx])
            prev_diff = curr_diff
        history_sum = [s + curr for s, curr in zip(history_sum, current_metrics)]
        history_count += 1
    return per_stride_diffs_i
