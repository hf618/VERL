# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
import ray
from ray.util.actor_pool import ActorPool
from .metrics_calculator import RepresentationMetricsCalculator
from .reward_manager_versions import RewardManager
from .reward_manager_versions import _default_compute_score
import time

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


import ray
import hydra

import numpy as np
import random
def set_global_seed(seed):
    """Set all relevant random number seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"The global random number seed has been set to:{seed}")

# This tells Hydra to use this function as the entry point and 
# to load the configuration from the config/ppo_trainer.yaml file (or a similar file).
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    seed = config.data.get('seed', 1)
    set_global_seed(seed)
    run_ppo(config, compute_score=_default_compute_score)


def run_ppo(config, compute_score=None):

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score=None):
    '''
    This is the core function that performs the PPO training.
    '''
    if config.trainer.get('plot_config', {}).get('enable', False):
        print("Plotting is enabled. Limiting PyTorch CPU threads to 1 on the driver to prevent deadlocks.")
        torch.set_num_threads(1)

    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # Purpose: This dictionary maps each role in the training process (e.g., ActorRollout, Critic, RewardModel) 
    # to the Ray worker class responsible for performing that role's computations.
    # Key: A Role enum member (e.g., Role.ActorRollout).
    # Value: A Ray remote class (e.g., ray.remote(ActorRolloutRefWorker)). 
    # This is the class that will be instantiated on the Ray cluster to perform the computations for that role.
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    
    calculator = RepresentationMetricsCalculator(tokenizer=tokenizer, 
                                                 compute_log_effective_rank=config.calculator.compute_log_effective_rank,
                                                 metric_indices=config.calculator.get('metric_indices', None),
                                                svd_rank=config.calculator.svd_rank, 
                                                 svd_niter=config.calculator.svd_niter,
                                                 zeroth_order_svd_method=config.calculator.zeroth_order_svd_method, 
                                                 diff_svd_method=config.calculator.diff_svd_method,
                                                 diff_calculator_method=config.calculator.diff_calculator_method
                                        )


    reward_fn = RewardManager(tokenizer=tokenizer, 
                              num_examine=0, 
                              compute_score=compute_score, 
                              calculator=calculator,
                              ema_alpha=config.reward_manager.ema_alpha,
                              indicator_names=config.reward_manager.indicator_names,
                              weights=config.reward_manager.weights,
                              weights_exploit=config.reward_manager.weights_exploit,
                              calculator_enabled=config.calculator.enable,
                              add_reward=config.reward_manager.add_reward,
                              modulation_gain=config.reward_manager.modulation_gain,
                              adv_estimator=config.algorithm.adv_estimator,
                              aux_reward_global_weight=config.reward_manager.aux_reward_global_weight,
                              aux_fix=config.reward_manager.aux_fix,
                              hypothesis_type=config.reward_manager.hypothesis_type
                            )
    
    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, 
                                  num_examine=1, 
                                  compute_score=None, 
                                  calculator=calculator,
                                  ema_alpha=config.reward_manager.ema_alpha,
                                  indicator_names=config.reward_manager.indicator_names,
                                  weights=config.reward_manager.weights,
                                  weights_exploit=config.reward_manager.weights_exploit,
                                  calculator_enabled=config.calculator.enable, 
                                  add_reward=config.reward_manager.add_reward,
                                  modulation_gain=config.reward_manager.modulation_gain,
                                    adv_estimator=config.algorithm.adv_estimator,
                                    aux_reward_global_weight=config.reward_manager.aux_reward_global_weight,
                                    aux_fix=config.reward_manager.aux_fix,
                                    hypothesis_type=config.reward_manager.hypothesis_type
                                  )


    # Purpose: This class manages the resource pools available on the Ray cluster and assigns roles to specific resource pools.
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            calculator=calculator)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
