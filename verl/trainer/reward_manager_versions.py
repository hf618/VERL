from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import kk
# from verl.utils.reward_score import simplelr_math
# from verl.utils.reward_score import deepseek_r1
from verl.utils.reward_score import hf_math_verify
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assume that _default_compute_score is accessible or has been imported in this file
# If it is in main_ppo.py, you need to move it here as well or to another common tool file
def _default_compute_score(data_source, solution_str, ground_truth):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    # elif data_source.lower() == "simplelr_math500" or data_source.lower() == "simplelr_aime24":
    #     return hf_math_verify.compute_accuracy(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    
    elif "kk" in data_source:
        return kk.compute_score(solution_str, ground_truth)
    elif "simplelr" in data_source:
        return hf_math_verify.compute_score(solution_str, ground_truth)
    elif "deepseek_r1" in data_source:
        return deepseek_r1.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
    

class RewardManager():
    """
    Optimized version based on the user's final request:
    1. Uses the nuanced 'score' for both the direct reward (reward_tensor_0)
       and for updating the performance EMA that controls the global scaling factor.
    2. Includes normalization to handle the [-1, 1] range of the score.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, calculator=None,
                 ema_alpha=0.7,
                 indicator_names=None,
                 weights=None,
                 weights_exploit=None,
                 calculator_enabled=True,
                 add_reward=True,
                 modulation_gain=1.5,
                 aux_reward_global_weight=1.0,
                 adv_estimator='grpo',
                 aux_fix=False,
                 hypothesis_type: str = "PlanB"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.calculator = calculator
        self.ema_alpha = ema_alpha
        self.indicator_names = indicator_names if indicator_names is not None else \
            ['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']
        
        self.weights_explore = weights if weights is not None else [0.0, 0.0, 1.0]
        self.weights_exploit = weights_exploit if weights_exploit is not None else [0.0, 1.0, 0.0]

        self.mids = {name: 0.0 for name in self.indicator_names}
        self.add_reward = add_reward
        self.calculator_enabled = calculator_enabled
        self.modulation_gain = modulation_gain
        self.epsilon = 1e-8
        
        # Tracks the EMA of the nuanced score
        # Initialized to 0.0, representing a neutral average score.
        self.ema_performance_score = 0.0 
        self.aux_reward_global_weight = aux_reward_global_weight
        self.aux_fix = aux_fix
        self.adv_estimator = adv_estimator
        self.hypothesis_type = hypothesis_type
    
        

    def __call__(self, data: DataProto, is_val=False, metrics_old=None, global_step=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # correctness_tensor = torch.zeros(len(data), dtype=torch.float32)
        reward_tensor_0 = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        correctness_tensor = torch.zeros(len(data), dtype=torch.bfloat16)
        calculator_tensor = torch.zeros(len(data), dtype=torch.bfloat16)
        already_print_data_sources = {}


        internal_metrics = {
            'percentage_deviation': [],
            'diff_2_tendency': [],
            'performance_scaling_factor': []

        }
        layer_key = '1'

        # It's only possible to calculate if it's enabled AND it's not the first step (metrics_old exists).
        use_aux_reward = self.add_reward and self.calculator_enabled and metrics_old

        performance_scaling_factor = 1.0 # Default scaling factor for step 1

        act_func = nn.Tanh() if use_aux_reward else None

        
        if use_aux_reward:
            if self.aux_fix:
                
                performance_scaling_factor = self.aux_reward_global_weight
                print(f"Using fixed performance_scaling_factor: {performance_scaling_factor}")
            else:
                
                normalized_performance = (self.ema_performance_score + 1.0) / 2.0
                performance_scaling_factor = self.aux_reward_global_weight * (1.0 - normalized_performance)
                print(f"Using dynamic performance_scaling_factor: {performance_scaling_factor}")
            internal_metrics['performance_scaling_factor'].append(performance_scaling_factor)

     
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']


            data_source = data_item.non_tensor_batch['data_source']

            score_dict = self.compute_score(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor_0[i, valid_response_length - 1] = score_dict['score']
            correctness_tensor[i] = score_dict['correctness']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)    



            reward_tensor[i, valid_response_length - 1] = reward_tensor_0[i, valid_response_length - 1]

            if use_aux_reward:
                # Calculate the 'Percentage Deviation' as the guidance signal
                guidance_indicator_name = self.indicator_names[0] # Diff 2
                current_guidance_value = data_item.batch['calculator_results'][layer_key][guidance_indicator_name]
                ema_baseline = self.mids[guidance_indicator_name]
                
                percentage_deviation = (current_guidance_value - ema_baseline) / (abs(ema_baseline) + self.epsilon)
                
                # We can still clamp this to prevent extreme values from having too much influence
                percentage_deviation = torch.clamp(percentage_deviation, -5.0, 5.0)


                
                # Interpolate between explore and exploit weight profiles
                w_explore = torch.tensor(self.weights_explore, device=data.batch.device)
                w_exploit = torch.tensor(self.weights_exploit, device=data.batch.device)


                if self.hypothesis_type == "PlanB":
                    
                    diff_2_tendency  = torch.sigmoid(self.modulation_gain * percentage_deviation)
                    dynamic_weights = diff_2_tendency  * w_explore + (1.0 - diff_2_tendency) * w_exploit
                    
                    internal_metrics['diff_2_tendency'].append(diff_2_tendency.item())
                elif self.hypothesis_type == "FixTwo":
                    
                    diff_2_tendency  = torch.tensor(0.5)
                    dynamic_weights = diff_2_tendency  * w_explore + (1.0 - diff_2_tendency) * w_exploit
                    
                    internal_metrics['diff_2_tendency'].append(diff_2_tendency.item())
                elif self.hypothesis_type == "FixOne":
                    
                    diff_2_tendency  = torch.tensor(0.0)
                    dynamic_weights = w_explore + w_exploit
                    
                    internal_metrics['diff_2_tendency'].append(diff_2_tendency.item())


                
                # Create a lookup for easier access
                weights_map = {name: weight for name, weight in zip(self.indicator_names, dynamic_weights)}

                
                internal_metrics['percentage_deviation'].append(percentage_deviation.item())
                internal_metrics['diff_2_tendency'].append(diff_2_tendency.item())
                
                for name, weight in weights_map.items():
                    log_name = f"weight_{name.replace(' ', '_').lower()}"
                    if log_name not in internal_metrics:
                        internal_metrics[log_name] = []
                    internal_metrics[log_name].append(weight.item())


                # sparse reward

                calculator_tensor_i = 0.0
                for indicator_name in self.indicator_names:
                    original_indicator = data_item.batch['calculator_results'][layer_key][indicator_name]
                    relative_deviation = (original_indicator - self.mids[indicator_name]) / (abs(self.mids[indicator_name]) + self.epsilon)
                    relative_deviation = torch.clamp(relative_deviation, -5.0, 5.0)

                    # Log the scalar relative deviation
                    log_name = f"relative_deviation_{indicator_name.replace(' ', '_').lower()}"
                    if log_name not in internal_metrics: internal_metrics[log_name] = []
                    internal_metrics[log_name].append(relative_deviation.item())
                    
                    calculator_tensor_i += act_func(relative_deviation) * weights_map[indicator_name]

                final_aux_reward = calculator_tensor_i * performance_scaling_factor
                reward_tensor[i, valid_response_length - 1] += final_aux_reward
        

        if use_aux_reward and not is_val:

            self.ema_performance_score = (1 - self.ema_alpha) * self.ema_performance_score + \
                                                self.ema_alpha * reward_tensor_0.sum(dim=-1).float().mean().cpu().item()
            
            for indicator_name in self.indicator_names:
                metric_key = f'cal/overall/layer_{layer_key}/{indicator_name}/mean'
                if metric_key in metrics_old:
                    v = metrics_old[metric_key]
                    self.mids[indicator_name] = (1 - self.ema_alpha) * self.mids[indicator_name] + self.ema_alpha * v

  
        return {"reward_tensor": reward_tensor, 
                "correctness_tensor": correctness_tensor, 
                "reward_tensor_0": reward_tensor_0,
                "internal_metrics": internal_metrics}


