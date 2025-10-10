# Training Configuration Details

This document provides a detailed explanation of the experiment settings available in the `train.sh` script.

## Experiment Settings

These variables are located at the end of the `train.sh` script and control the model, data, and training parameters.

```sh
bash train_grpo_math_tune_ray.sh \
    --model_name mistralai/Mathstral-7B-v0.1 --max_prompt_length 512 --max_response_length 1536 \
    --critic_model_path "" --adv_estimator "grpo"  \
    ...
```

### Parameter Descriptions

| Variable        | Description                                                                                             | Default Value                        |
|-----------------|---------------------------------------------------------------------------------------------------------|--------------------------------------|
| `adv_estimator` | Control the calculation of the advantage value. When set to "grpo", execute GRPO; when set to "gae", execute PPO. | `grpo`      |
| `add_reward`    | Whether to add auxiliary metrics to the original task reward.                                           | `False`      |
| `add_adv`       | Whether to add auxiliary metrics to the original advantage values.                                      | `True`               |
| `compute_global_metrics` | Whether to calculate the dataset-level zero-order metrics during training and validation (Just for observation).                   | `True`                               |
| `compute_cumulative_global_metrics`| Whether to calculate the dataset-level 1-order & 2-order metrics during training and validation (Just for observation).  | `True`                                  |
| `global_diff_stride_train` | Stride for dataset-level metrics during training (To ensure computational efficiency, it is necessary to consider your actual train_batch_size.)   | `20`   |
| `global_diff_stride_val`   | Stride for dataset-level metrics during validation (To ensure computational efficiency, it is necessary to consider your actual validation dataset size.) | `20`       |
| `enable_calculator`       | Whether to calculate the response-level metrics.                                      | `True`               |
| `metric_indices`       | Defines specific metrics for calculating response-level indicators, details please refer to [calculator.metric_indices](verl/trainer/config/ppo_trainer.yaml)      | `[1,2]`   |
| `reward_indicator_names`    | Metrics of auxiliary signals feedback to policy.  | `['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']` |
| `reward_weights`       | Exploration-focused vector ($\mathbf{w}_{\mathrm{explore}}$)      | `[0.0, 0.0, 1.0]`   |
| `reward_weights_exploit`       | Exploitation-focused vector ($\mathbf{w}_{\mathrm{exploit}}$)     | `[0.0, 1.0, 0.0]`   |
| `diff_stride`       | Stride for response-level metrics ($s$)      | `40`   |
| `aux_reward_global_weight`       | Total weight of auxiliary signals  | `1`   |
| `reward_ema_alpha`       | Exponential Moving Average (EMA) $\alpha$ for each metrics  | `1`   |
| `adv_shaping_kappa`       | Auxiliary advantage clipping factor  | `2`   |