
import torch
import torch.nn.functional as F
import os
import time
import ray
from ray.util.actor_pool import ActorPool
import numpy as np
from . import metrics_utils

class RepresentationMetricsCalculator():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, 
                 metric_indices=None, 
                 zeroth_order_svd_method: str = 'full',
                 diff_svd_method: str = 'lowrank',
                 svd_rank: int = 6,
                 svd_niter: int = 5,
                 compute_log_effective_rank: bool = False,
                 compute_global_metrics: bool = False,
                 compute_cumulative_global_metrics: bool = False,
                 diff_calculator_method: str = 'optimized' 
                 ):
        """
        Initializes the RepresentationMetricsCalculator.

        Args:
            tokenizer: The tokenizer object (not directly used in metric calculation, but for context).
            max_seq_len (int): Maximum sequence length to process for memory optimization. Defaults to 512.
            compute_log_effective_rank (bool): If True, calculates and includes the log of Effective Rank and its differences. Defaults to False.
            svd_rank (int): The rank of low-rank SVD
            zeroth_order_svd_method (str): SVD method for 0-order metrics ('full' or 'lowrank').
            diff_svd_method (str): SVD method for 1st/2nd order diffs ('full' or 'lowrank').
            svd_rank (int): The rank for low-rank SVD calculations.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.epsilon = 1e-8
        self.compute_log_effective_rank = compute_log_effective_rank # New flag for log effective rank
        self.compute_global_metrics = compute_global_metrics
        self.compute_cumulative_global_metrics = compute_cumulative_global_metrics
        

        self._cached_tensors = {}
        self.zeroth_order_svd_method = zeroth_order_svd_method
        self.diff_svd_method = diff_svd_method
        self.svd_rank = svd_rank
        self.svd_niter = svd_niter

        self.diff_calculator_method = diff_calculator_method

        # Define all available basic indicators and their calculation functions
        all_base_metrics = [
            ("Response Entropy 1", self.calculate_response_entropy),
            ("Effective Rank", lambda hs, mask: self.calculate_effective_rank(hs, mask, log_output=False)),
            ("Traditional Rank", self.calculate_traditional_rank),
            ("Curvature", self.calculate_curvature)
        ]

        # Dynamically add Log Effective Rank if needed
        if self.compute_log_effective_rank:
            all_base_metrics.append(
                ("Log Effective Rank", lambda hs, mask: self.calculate_effective_rank(hs, mask, log_output=True))
            )
        

        if metric_indices is None:

            self.selected_metrics = all_base_metrics
        else:

            self.selected_metrics = [all_base_metrics[i] for i in metric_indices if i < len(all_base_metrics)]
        
        print(f"[RepresentationMetricsCalculator] Initialized with selected metrics: {[name for name, _ in self.selected_metrics]}")

    def __call__(self, hidden_states, attention_mask, compute_diff=False, diff_stride=1):
        with torch.inference_mode():
            batch_size, seq_len, num_layers, hidden_dim = hidden_states.shape
            results = {}
            
            for layer_idx in range(num_layers):
                layer_key = str(layer_idx + 1)
                layer_hidden = hidden_states[:, :, layer_idx, :].contiguous()

                # Compute all sequence-level metrics as usual
                base_metrics = {
                    name: func(layer_hidden, attention_mask)
                    for name, func in self.selected_metrics
                }
                
                per_stride_diffs = {}
                if compute_diff:
                    final_diffs, per_stride_diffs = self.calculate_metric_diff(
                        layer_hidden, attention_mask, diff_stride
                    )
                    base_metrics.update(final_diffs)
                
                results[layer_key] = base_metrics
                self._free_memory()
                
            return results

    def _aggregate_diffs(self, all_per_stride_diffs, batch_size, device, selected_metric_names):
        """Auxiliary function: Aggregate the final diff tensors from per-stride results."""
        final_diffs = {f"{name} diff": torch.zeros(batch_size, device=device) for name in selected_metric_names}
        final_diffs.update({f"{name} diff 2": torch.zeros(batch_size, device=device) for name in selected_metric_names})
        
        for i in range(batch_size):
            per_stride_diffs_i = all_per_stride_diffs[i]
            for name in selected_metric_names:
                diff_key = f"{name} diff"
                if diff_key in per_stride_diffs_i and per_stride_diffs_i[diff_key]:
                    final_diffs[diff_key][i] = torch.tensor(per_stride_diffs_i[diff_key]).mean()
                
                diff2_key = f"{name} diff 2"
                if diff2_key in per_stride_diffs_i and per_stride_diffs_i[diff2_key]:
                    final_diffs[diff2_key][i] = torch.tensor(per_stride_diffs_i[diff2_key]).mean()

        return final_diffs
    
    def calculate_aggregated_metrics(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Computes metrics on a globally aggregated matrix of mean hidden states.

        Args:
            hidden_states (torch.Tensor): The full hidden states tensor for the batch 
                                          (batch_size, seq_len, num_layers, hidden_dim).
            attention_mask (torch.Tensor): The attention mask for the response part 
                                           (batch_size, seq_len).

        Returns:
            dict: A dictionary containing the computed global metrics for each layer.
                  e.g., {"1": {"global/Effective Rank": 15.7, ...}, "2": {...}}
        """
        with torch.inference_mode():
            batch_size, seq_len, num_layers, hidden_dim = hidden_states.shape
            device = hidden_states.device
            global_results = {}

            print("Computing global aggregated metrics...")

            for layer_idx in range(num_layers):
                layer_key = str(layer_idx + 1)
                layer_hidden = hidden_states[:, :, layer_idx, :].contiguous()
                

                aggregated_matrix = torch.zeros(batch_size, hidden_dim, device=device, dtype=layer_hidden.dtype)


                for i in range(batch_size):
                    mask = attention_mask[i].bool()
                    valid_hidden = layer_hidden[i, mask, :]
                    if valid_hidden.shape[0] > 0:
                        aggregated_matrix[i] = valid_hidden.mean(dim=0)
                

                layer_global_metrics = {}
                for name, func in self.selected_metrics:
 
                    metric_value = func(aggregated_matrix.unsqueeze(0), None) #unsqueeze(0) to make it (1, num_samples, hidden_dim)
                    layer_global_metrics[f"global/{name}"] = metric_value.item()

                global_results[layer_key] = layer_global_metrics
            
            return global_results
 
    def calculate_metric_diff(self, hidden_states, attention_mask, stride):
        batch_size, _, _ = hidden_states.shape
        selected_metric_names = [name for name, _ in self.selected_metrics]
        
        all_per_stride_diffs = []


        if self.diff_calculator_method == 'optimized':
            target_func = metrics_utils.calculate_diffs_for_single_sample_optimized
        else:
            target_func = metrics_utils.calculate_diffs_for_single_sample_original

        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            
            if valid_hidden.size(0) < 2:
          
                empty_result = {f"{name} diff": [] for name in selected_metric_names}
                empty_result.update({f"{name} diff 2": [] for name in selected_metric_names})
                all_per_stride_diffs.append(empty_result)
                continue

            per_stride_diffs_i = target_func(
                valid_hidden, self.max_seq_len, stride, selected_metric_names, 
                self.svd_rank, self.svd_niter, self.diff_svd_method
            )
            all_per_stride_diffs.append(per_stride_diffs_i)
            
    
        device = hidden_states.device
        final_diffs = self._aggregate_diffs(all_per_stride_diffs, batch_size, device, selected_metric_names)

        per_stride_diffs = {f"{name} diff": [[] for _ in range(batch_size)] for name in selected_metric_names}
        per_stride_diffs.update({f"{name} diff 2": [[] for _ in range(batch_size)] for name in selected_metric_names})
        for i in range(batch_size):
            for key in per_stride_diffs.keys():
                if key in all_per_stride_diffs[i]:
                    per_stride_diffs[key][i] = all_per_stride_diffs[i][key]

        return final_diffs, per_stride_diffs
    

    def _free_tensors(self, tensors):
        """
        Explicitly frees a list of PyTorch tensors from memory.

        Args:
            tensors (list): A list of torch.Tensor objects to be deleted.
        """
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        # Clear CUDA cache to release GPU memory (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _free_memory(self):
        """
        Clears the internal cache and explicitly frees memory.
        This is called periodically to manage memory usage.
        """
        self._cached_tensors.clear() # Clear the cache of intermediate results
        self._free_tensors([]) # Call _free_tensors with an empty list to just clear CUDA cache
    
    def calculate_response_entropy(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'covariance') -> torch.Tensor:
        """
        Calculates Renyi entropy for each sample in a batch.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            alpha (float): The alpha parameter for Renyi entropy. Defaults to 1.0001.
            matrix_type (str): Type of matrix to use, 'covariance' or 'gram'. Defaults to 'covariance'.

        Returns:
            torch.Tensor: A tensor of Renyi entropies for each sample in the batch.
        """
        assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        entropies = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # Extract non-padding tokens
            
            entropies[i] = metrics_utils.compute_single_entropy(valid_hidden, alpha, matrix_type)
        return entropies
    
    def _calculate_and_cache_ranks(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, log_output: bool):
        """Internal function used to calculate and cache the effective rank and traditional rank, ensuring that SVD runs only once"""
        cache_key = (id(hidden_states), log_output)
        if cache_key in self._cached_tensors:
            return self._cached_tensors[cache_key]

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        effective_ranks = torch.zeros(batch_size, device=device, dtype=torch.bfloat16)
        traditional_ranks = torch.zeros(batch_size, device=device, dtype=torch.bfloat16)


        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            
            eff_rank, trad_rank = metrics_utils.compute_single_effective_rank(
                valid_hidden, self.svd_rank, self.svd_niter, log_output, self.zeroth_order_svd_method
            )
            effective_ranks[i] = eff_rank
            traditional_ranks[i] = trad_rank
        
        self._cached_tensors[cache_key] = (effective_ranks, traditional_ranks)
        return effective_ranks, traditional_ranks
    
    def calculate_effective_rank(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, log_output: bool = False) -> torch.Tensor:
        effective_ranks, _ = self._calculate_and_cache_ranks(hidden_states, attention_mask, log_output)
        return effective_ranks
    
    def calculate_traditional_rank(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, traditional_ranks = self._calculate_and_cache_ranks(hidden_states, attention_mask, log_output=False)
        return traditional_ranks
      
    def calculate_curvature(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates average curvature for each sample in a batch by calling the single-sample helper.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: A tensor of average curvatures for each sample in the batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        curvatures = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
     
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            

            curvatures[i] = metrics_utils.compute_single_curvature(valid_hidden)
        return curvatures
    

    


