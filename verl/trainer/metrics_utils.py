import time
import threading
import torch
import torch.nn.functional as F


# ============================================================
# 0. Const
# ============================================================

EPS = 1e-8

# ============================================================
# 1. Batched Metric Helpers (Tensor-based)
# ============================================================

def batched_entropy(eigvals: torch.Tensor, alpha: float = 1.0001) -> torch.Tensor:
    """
    Computes Renyi/Shannon entropy for a batch of eigenvalues.
    eigvals: [B, N]
    """
    val_sum = eigvals.sum(dim=-1, keepdim=True) + EPS
    p = eigvals / val_sum

    if abs(alpha - 1.0) < 1e-6:
        p_safe = p + EPS
        H = -(p * torch.log(p_safe)).sum(dim=-1)
    else:
        sum_p_alpha = (p ** alpha).sum(dim=-1)
        H = (1.0 / (1.0 - alpha)) * torch.log(sum_p_alpha + EPS)

    return H  # [B]

def batched_effective_rank(eigvals: torch.Tensor, log_output: bool = False) -> torch.Tensor:
    """
    eigvals: [B, N]
    Returns: [B]
    """
    S = torch.sqrt(torch.relu(eigvals))
    S_sum = S.sum(dim=-1, keepdim=True) + EPS
    p = S / S_sum
    H = -(p * torch.log(p + EPS)).sum(dim=-1)

    if log_output:
        return H
    else:
        return torch.exp(H)
    
def batched_traditional_rank(eigvals: torch.Tensor, ref_dim: int) -> torch.Tensor:
    """
    eigvals: [B, N]
    ref_dim: Used for tolerance calculation
    """
    S = torch.sqrt(torch.relu(eigvals))
    max_S = S.max(dim=-1, keepdim=True).values
    tol = max_S * ref_dim * torch.finfo(S.dtype).eps
    return (S > tol).sum(dim=-1).to(torch.float32)

def compute_single_curvature(hidden: torch.Tensor) -> float:
    """Calculate the curvature of a single sample"""
    if hidden.size(0) < 3:
        return 0.0

    diffs = hidden[1:] - hidden[:-1]
    norms = torch.norm(diffs, dim=1, keepdim=True)
    valid_mask = (norms > 1e-6).squeeze(-1)
    valid_diffs = diffs[valid_mask]

    if valid_diffs.size(0) < 2:
        return 0.0

    v1 = valid_diffs[:-1]
    v2 = valid_diffs[1:]
    cos_sim = F.cosine_similarity(v1, v2, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.arccos(cos_sim)

    if angles.numel() == 0:
        return 0.0

    return angles.mean().item()

# ============================================================
# 2. Single Sample Legacy Wrappers (For compatibility)
# ============================================================

def compute_single_entropy(
    hidden: torch.Tensor,
    alpha: float = 1.0001,
    matrix_type: str = "gram",
) -> float:
    if hidden.size(0) < 2:
        return 0.0
    try:
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)
        L, D = centered.shape
        if L < D:
            G = centered @ centered.T
        else:
            G = centered.T @ centered

        eigvals = torch.linalg.eigvalsh(G)
        k = min(L, D)
        eigvals = eigvals[-k:]
        eigvals = eigvals[eigvals > 1e-8]
        if len(eigvals) == 0:
            return 0.0

        normalized = eigvals / eigvals.sum()
        if abs(alpha - 1.0) < 1e-6:
            normalized = normalized[normalized > 1e-12]
            return -torch.sum(normalized * torch.log(normalized)).item()
        else:
            return (1 / (1 - alpha)) * torch.log(torch.sum(normalized ** alpha)).item()
    except Exception:
        return 0.0
    
def compute_single_effective_rank(
    hidden: torch.Tensor,
    svd_rank: int,
    svd_niter: int,
    log_output: bool = False,
    method: str = "lowrank",
) -> tuple[float, float]:
    if hidden.size(0) < 2:
        return 0.0, 0.0
    try:
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)
        L, D = centered.shape
        if L < D:
            G = centered @ centered.T
        else:
            G = centered.T @ centered

        eigvals = torch.linalg.eigvalsh(G)
        k = min(L, D)
        eigvals = eigvals[-k:]

        S = torch.sqrt(torch.relu(eigvals))
        if S.numel() == 0:
            return 0.0, 0.0

        tol = S.max() * max(L, D) * torch.finfo(S.dtype).eps
        trad_rank = torch.sum(S > tol).item()

        S_sum = S.sum() + EPS
        normalized_S = S / S_sum
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + EPS)).item()

        eff_rank = entropy if log_output else torch.exp(torch.tensor(entropy)).item()
        return eff_rank, float(trad_rank)
    except Exception:
        return 0.0, 0.0
    
# ============================================================
# 3. Single-device batched core (集成老哥 batched Gram + 改进)
# ============================================================

def _calculate_diffs_batched_single_device(
    hidden_states,
    attention_mask,
    max_seq_len,
    stride,
    selected_metric_names,
    svd_rank,
    svd_niter,
    svd_method,
    base_metrics_batch=None,
):
    """
    单卡上的 batched diff 计算核心：
    - 利用 masked hidden + Gram 矩阵 + chunking
    - k_t < D 时用 L-mode (HH^T)，k_t >= D 时用 D-mode (H^T H)
    - 支持 Response Entropy / Effective Rank / Log Effective Rank / Traditional Rank / Curvature
    """
    t0_total = time.perf_counter()

    batch_size, total_seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    # 有效长度由 attention_mask 决定
    lengths = attention_mask.sum(dim=1).long()
    if max_seq_len is not None:
        max_seq_tensor = torch.tensor(max_seq_len, device=device, dtype=torch.long)
        lengths = torch.min(lengths, max_seq_tensor)
        K_max = min(total_seq_len, max_seq_len)
    else:
        K_max = total_seq_len

    if K_max < 2 or stride <= 0:
        # 太短或者 stride 不合法，直接返回空结果
        results_storage = {
            f"{name} diff": [torch.tensor([], device=device) for _ in range(batch_size)]
            for name in selected_metric_names
        }
        results_storage.update(
            {
                f"{name} diff 2": [torch.tensor([], device=device) for _ in range(batch_size)]
                for name in selected_metric_names
            }
        )
        return 0.0, results_storage

    # 用 attention_mask 把 padding hidden 清零
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
    hidden = hidden_states * mask_expanded
    hidden = hidden[:, :K_max, :].contiguous()  # [B, K_max, D]

    # 输出存储：每个 metric 一个 [B] 的 list，每个元素里再 append scalar
    results_storage = {
        f"{name} diff": [[] for _ in range(batch_size)] for name in selected_metric_names
    }
    results_storage.update(
        {f"{name} diff 2": [[] for _ in range(batch_size)] for name in selected_metric_names}
    )

    # 历史统计（每个 sample / 每个 metric）
    n_metrics = len(selected_metric_names)
    history_sums = torch.zeros(batch_size, n_metrics, device=device, dtype=torch.float32)
    history_counts = torch.zeros(batch_size, device=device, dtype=torch.float32)
    prev_diffs = torch.zeros(batch_size, n_metrics, device=device, dtype=torch.float32)
    has_prev_diff = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 记录每个样本最后一次 stride 的 k_t，用来决定要不要复用 base_metrics
    last_stride_t = torch.zeros(batch_size, dtype=torch.long, device=device)

    metric_map = {name: i for i, name in enumerate(selected_metric_names)}
    has_curvature = "Curvature" in selected_metric_names
    curv_idx = metric_map["Curvature"] if has_curvature else -1

    # 为了控制 D×D Gram 的显存，按 batch 维度切 chunk
    chunk_size = 32
    total_compute_time = 0.0

    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        h_chunk = hidden[start:end]        # [B_chunk, K_max, D]
        len_chunk = lengths[start:end]     # [B_chunk]
        B_chunk = h_chunk.size(0)

        hist_sum_chunk = history_sums[start:end]
        hist_cnt_chunk = history_counts[start:end]
        prev_diff_chunk = prev_diffs[start:end]
        has_prev_chunk = has_prev_diff[start:end]

        t0_chunk = time.perf_counter()

        # 对该 chunk 的所有 stride 进行循环
        for k_t in range(stride, K_max + 1, stride):
            # 只对长度 >= k_t 的样本算这一档
            active_mask = len_chunk >= k_t
            if not active_mask.any():
                continue

            # 当前窗口 [0, k_t)
            H_t = h_chunk[:, :k_t, :]  # [B_chunk, k_t, D]
            # eigvalsh on CPU does not support bfloat16; upcast for stability
            if H_t.dtype != torch.float32:
                H_t = H_t.float()
            # 做中心化：Hc = H - mean_t
            mu = H_t.mean(dim=1, keepdim=True)
            Hc = H_t - mu  # [B_chunk, k_t, D]

            # L-mode / D-mode 选择
            if k_t < hidden_dim:
                # L-mode：Gram_L = Hc Hc^T，[B_chunk, k_t, k_t]
                G = torch.matmul(Hc, Hc.transpose(1, 2))
            else:
                # D-mode：Gram_D = Hc^T Hc，[B_chunk, D, D]
                H_T = Hc.transpose(1, 2)  # [B_chunk, D, k_t]
                G = torch.matmul(H_T, Hc)  # [B_chunk, D, D]

            # 求特征值：形状为 [B_chunk, N]，N = k_t or hidden_dim
            eigvals = torch.linalg.eigvalsh(G)

            # 当前 stride 下的所有 metrics，shape [B_chunk, n_metrics]
            curr_metrics_vals = torch.zeros(
                B_chunk, n_metrics, device=device, dtype=torch.float32
            )

            # 先把非 Curvature 的 metric 批量算好
            for name, idx in metric_map.items():
                if name == "Effective Rank":
                    curr_metrics_vals[:, idx] = batched_effective_rank(
                        eigvals, log_output=False
                    )
                elif name == "Log Effective Rank":
                    curr_metrics_vals[:, idx] = batched_effective_rank(
                        eigvals, log_output=True
                    )
                elif name == "Traditional Rank":
                    # ref_dim 用 hidden_dim 保持容差尺度一致
                    curr_metrics_vals[:, idx] = batched_traditional_rank(
                        eigvals, ref_dim=hidden_dim
                    )
                elif name == "Response Entropy 1":
                    curr_metrics_vals[:, idx] = batched_entropy(
                        eigvals, alpha=1.0001
                    )
                # Curvature 下面再算

            # Curvature 按 active 样本 loop 算（用未中心化的 H_t）
            if has_curvature:
                active_indices = torch.nonzero(active_mask).squeeze(-1)
                for loc_idx in active_indices.tolist():
                    # H_t[loc_idx] 是 [k_t, D]
                    val = compute_single_curvature(H_t[loc_idx])
                    curr_metrics_vals[loc_idx, curv_idx] = val

            # 更新历史，并写入 diff / diff2
            active_indices = torch.nonzero(active_mask).squeeze(-1)
            if active_indices.numel() > 0:
                for loc_idx in active_indices.tolist():
                    g_idx = start + loc_idx  # 全局 batch idx
                    curr_vals = curr_metrics_vals[loc_idx]  # [n_metrics]

                    # 当前这一步之前已有多少历史步
                    cnt_prev = hist_cnt_chunk[loc_idx].item()

                    if cnt_prev > 0:
                        avg_prev = hist_sum_chunk[loc_idx] / (cnt_prev + EPS)
                        diff = curr_vals - avg_prev  # [n_metrics]

                        # 存 diff
                        for m_idx, m_name in enumerate(selected_metric_names):
                            results_storage[f"{m_name} diff"][g_idx].append(
                                float(diff[m_idx].item())
                            )

                        # 存 diff2
                        if has_prev_chunk[loc_idx]:
                            diff2 = diff - prev_diff_chunk[loc_idx]
                            for m_idx, m_name in enumerate(selected_metric_names):
                                results_storage[f"{m_name} diff 2"][g_idx].append(
                                    float(diff2[m_idx].item())
                                )

                        prev_diff_chunk[loc_idx] = diff
                        has_prev_chunk[loc_idx] = True

                    # 更新历史统计
                    hist_sum_chunk[loc_idx] += curr_vals
                    hist_cnt_chunk[loc_idx] += 1.0
                    last_stride_t[g_idx] = k_t

        t1_chunk = time.perf_counter()
        total_compute_time += (t1_chunk - t0_chunk)

    # 把 chunk 视图写回全局（虽然是 view，本质已经同步，这里只是显式强调）
    history_sums[:] = history_sums
    history_counts[:] = history_counts
    prev_diffs[:] = prev_diffs
    has_prev_diff[:] = has_prev_diff

    # 4. 复用 base_metrics_batch（只在没有走到最后长度的时候用）
    if base_metrics_batch is not None:
        for i in range(batch_size):
            # 是否有完整的 base metrics
            all_found = True
            base_vals = []
            for name in selected_metric_names:
                if (
                    name in base_metrics_batch
                    and base_metrics_batch[name] is not None
                    and base_metrics_batch[name].numel() > i
                ):
                    base_vals.append(base_metrics_batch[name][i].item())
                else:
                    all_found = False
                    break

            if not all_found:
                continue

            # 如果 stride 已经覆盖到整个有效长度（或超过），就没必要再补一次
            if last_stride_t[i].item() >= max(1, lengths[i].item() - 1):
                continue

            cnt = history_counts[i].item()
            if cnt <= 0:
                continue

            vals_t = torch.tensor(base_vals, device=device, dtype=torch.float32)
            avg_t = history_sums[i] / (cnt + EPS)
            diff_t = vals_t - avg_t

            # 存 diff
            for m_idx, name in enumerate(selected_metric_names):
                results_storage[f"{name} diff"][i].append(float(diff_t[m_idx].item()))

            # 存 diff2
            if has_prev_diff[i]:
                diff2_t = diff_t - prev_diffs[i]
                for m_idx, name in enumerate(selected_metric_names):
                    results_storage[f"{name} diff 2"][i].append(
                        float(diff2_t[m_idx].item())
                    )

    # 5. 把 list -> tensor
    final_results: dict[str, list[torch.Tensor]] = {}
    for key, list_of_lists in results_storage.items():
        final_results[key] = [
            torch.tensor(l, device=device, dtype=torch.float32) for l in list_of_lists
        ]

    total_time = time.perf_counter() - t0_total
    return total_time, final_results


# ============================================================
# 4. Multi-GPU wrapper（自动按 batch 维度切到多卡）
# ============================================================

def calculate_diffs_batched(
    hidden_states,
    attention_mask,
    max_seq_len,
    stride,
    selected_metric_names,
    svd_rank,
    svd_niter,
    svd_method,
    base_metrics_batch=None,
):
    """
    Multi-GPU aware wrapper around the single-device implementation.

    使用方式：
    - 如果 torch.cuda.device_count() <= 1：直接单卡跑 _calculate_diffs_batched_single_device
    - 如果有多张可见 GPU（由 CUDA_VISIBLE_DEVICES 控制），自动把 batch 切块分到
      cuda:0, cuda:1, ... 上，并行计算 diff，最后聚合。
    """
    
    # 没 GPU 或只有一张，直接单卡
    if (not torch.cuda.is_available()) or (torch.cuda.device_count() <= 1):
        return _calculate_diffs_batched_single_device(
            hidden_states,
            attention_mask,
            max_seq_len,
            stride,
            selected_metric_names,
            svd_rank,
            svd_niter,
            svd_method,
            base_metrics_batch=base_metrics_batch,
        )

    gpu_ids = list(range(torch.cuda.device_count()))
    batch_size = hidden_states.shape[0]

    # batch 太小也没必要并行
    if batch_size <= 1:
        return _calculate_diffs_batched_single_device(
            hidden_states,
            attention_mask,
            max_seq_len,
            stride,
            selected_metric_names,
            svd_rank,
            svd_niter,
            svd_method,
            base_metrics_batch=base_metrics_batch,
        )

    device_orig = hidden_states.device

    # 准备全局结果：每个 metric 对应一个长度为 B 的 list，先填 None
    all_keys = [f"{n} diff" for n in selected_metric_names] + [
        f"{n} diff 2" for n in selected_metric_names
    ]
    global_results: dict[str, list[torch.Tensor | None]] = {
        k: [None] * batch_size for k in all_keys
    }

    # 把样本索引均匀分给各 GPU
    indices = list(range(batch_size))
    n_gpu = len(gpu_ids)
    per_gpu_bs = (batch_size + n_gpu - 1) // n_gpu

    index_chunks: list[list[int]] = []
    for i in range(n_gpu):
        s = i * per_gpu_bs
        e = min(batch_size, (i + 1) * per_gpu_bs)
        if s >= e:
            break
        index_chunks.append(indices[s:e])
    gpu_ids = gpu_ids[: len(index_chunks)]

    times = [0.0 for _ in range(len(index_chunks))]
    threads: list[threading.Thread] = []

    def worker(worker_idx: int, dev_id: int, idx_list: list[int]):
        if not idx_list:
            return

        local_device = torch.device(f"cuda:{dev_id}")

        hs_local = hidden_states[idx_list].to(local_device)
        am_local = attention_mask[idx_list].to(local_device)

        base_local = None
        if base_metrics_batch is not None:
            base_local = {}
            for name, tensor in base_metrics_batch.items():
                if tensor is None:
                    continue
                base_local[name] = tensor[idx_list].to(local_device)

        local_time, local_res = _calculate_diffs_batched_single_device(
            hs_local,
            am_local,
            max_seq_len,
            stride,
            selected_metric_names,
            svd_rank,
            svd_niter,
            svd_method,
            base_metrics_batch=base_local,
        )

        times[worker_idx] = float(local_time)

        # 把子结果搬回原始 device，并填到 global_results 对应位置
        for key, tensor_list in local_res.items():
            if key not in global_results:
                continue
            assert len(tensor_list) == len(idx_list)
            for j, global_idx in enumerate(idx_list):
                global_results[key][global_idx] = tensor_list[j].to(device_orig)

    # 每个 GPU 启一个线程
    for w_idx, (dev_id, idx_list) in enumerate(zip(gpu_ids, index_chunks)):
        t = threading.Thread(target=worker, args=(w_idx, dev_id, idx_list))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # 防御性：如果有样本没被算到，用单卡再补
    missing_indices = [
        i
        for i in range(batch_size)
        if any(global_results[k][i] is None for k in global_results)
    ]
    if missing_indices:
        hs_local = hidden_states[missing_indices].to(device_orig)
        am_local = attention_mask[missing_indices].to(device_orig)
        base_local = None
        if base_metrics_batch is not None:
            base_local = {
                name: tensor[missing_indices].to(device_orig)
                for name, tensor in base_metrics_batch.items()
                if tensor is not None
            }
        extra_time, extra_res = _calculate_diffs_batched_single_device(
            hs_local,
            am_local,
            max_seq_len,
            stride,
            selected_metric_names,
            svd_rank,
            svd_niter,
            svd_method,
            base_metrics_batch=base_local,
        )
        times.append(float(extra_time))
        for key, tensor_list in extra_res.items():
            if key not in global_results:
                continue
            assert len(tensor_list) == len(missing_indices)
            for j, global_idx in enumerate(missing_indices):
                global_results[key][global_idx] = tensor_list[j].to(device_orig)

    # 把 None 填成空 tensor（理论上不应该出现）
    for key, per_batch_list in global_results.items():
        for i in range(batch_size):
            if per_batch_list[i] is None:
                per_batch_list[i] = torch.tensor(
                    [], device=device_orig, dtype=torch.float32
                )

    total_shared_time = max(times) if times else 0.0
    return total_shared_time, global_results


def _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names, max_dim=None):
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
                    
                    ref_dim = max_dim if max_dim is not None else max(S.shape)
                    tol = S.max() * ref_dim * torch.finfo(S.dtype).eps
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
                                                svd_rank, svd_niter, svd_method,
                                                final_base_metrics=None): # <--- [新增参数]
    """
    [Final Adaptive Production Version with Reuse Optimization]
    """
    valid_len = valid_hidden.size(0)
    hidden_dim = valid_hidden.size(1)
    
    # 检查是否是全量模式，如果是截断模式且序列过长，则不能复用 Base Metric (因为 Base 是全量的，而这里是局部的)
    is_full_sequence_window = (valid_len <= max_seq_len)

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
    
    last_t = 0 

    for t in range(stride, valid_len, stride):
        current_window = valid_hidden[:t+1]
        
        # A. Shared Computation
        t0_shared = time.perf_counter()
        current_window_f32 = current_window.to(torch.float32)
        
        new_chunk = current_window_f32[processed_idx:]
        s = s + new_chunk.sum(dim=0, keepdim=True)
        k = current_window_f32.shape[0]


        current_max_dim = max(k, current_window_f32.shape[1])

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
        
        current_metrics = _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names, max_dim=current_max_dim)
        
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

    # === [新增] Final Step Reuse Logic ===
    # 条件：用户传入了 Base Metrics，且没有发生窗口截断（或者 Base Metrics 本身就是针对当前窗口的）
    # 你的场景是 max_seq_len = max_response_length，所以 is_full_sequence_window 应该为 True
    if final_base_metrics is not None and is_full_sequence_window:
        # 构造最后一步的 Metrics 列表
        final_metrics_vec = []
        all_found = True
        for name in selected_metric_names:
            if name in final_base_metrics:
                final_metrics_vec.append(final_base_metrics[name])
            else:
                # 如果某个指标（比如 Curvature）在 Base Metrics 里没算，就没法复用了
                all_found = False
                break
        
        # 只有当循环最后一步确实没走到终点 (防止重复计算)，并且所有指标都能复用时执行
        if all_found and (last_t < valid_len - 1):
            # 执行标准的 Diff 更新逻辑 (纯数学加减，耗时几乎为0)
            if history_count > 0:
                hist_avg = [sm / history_count for sm in history_sum]
                curr_diff = [(curr - avg) for curr, avg in zip(final_metrics_vec, hist_avg)]
                
                for idx, name in enumerate(selected_metric_names): 
                    per_stride_diffs_i[f"{name} diff"].append(curr_diff[idx])
                    # 这里不需要加 timing，因为是免费复用的
                
                if prev_diff is not None:
                    curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                    for idx, name in enumerate(selected_metric_names): 
                        per_stride_diffs_i[f"{name} diff 2"].append(curr_diff2[idx])
            
            # 不需要更新 history_sum 了，因为这是最后一步
            
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
