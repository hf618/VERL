import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
from collections import Counter


def _calculate_pairwise_euclidean_dist(embeddings: torch.Tensor) -> float:
    """Calculate the average pairwise Euclidean distance between the mean representations of all rollouts"""
    if embeddings.shape[0] < 2:
        return 0.0
    embeddings_f32 = embeddings.to(torch.float32)
    return torch.pdist(embeddings_f32, p=2).mean().item()

def _calculate_pairwise_cosine_sim(embeddings: torch.Tensor) -> float:
    """Calculate the average pairwise cosine similarity between the mean representations of all rollouts"""
    if embeddings.shape[0] < 2:
        return 0.0
    embeddings_f32 = embeddings.to(torch.float32)
    n = embeddings_f32.shape[0]
    norm_embeds = F.normalize(embeddings_f32, p=2, dim=1)
    sim_matrix = norm_embeds @ norm_embeds.T
    triu_indices = torch.triu_indices(n, n, offset=1)
    return sim_matrix[triu_indices[0], triu_indices[1]].mean().item()

def _calculate_dist_to_center(embeddings: torch.Tensor) -> float:
    """Calculate the average distance between the mean representation of each rollout and its center point"""
    if embeddings.shape[0] < 1:
        return 0.0
    embeddings_f32 = embeddings.to(torch.float32)
    center = embeddings_f32.mean(dim=0)
    distances = torch.norm(embeddings_f32 - center, p=2, dim=1)
    return distances.mean().item()

def _calculate_pairwise_token_jsd(p1_distributions: torch.Tensor, rollout_n: int) -> float:
    """Calculate the average pairwise JSD between all rollouts' token distributions"""
    if rollout_n < 2:
        return 0.0
    P = p1_distributions.to(torch.float32)
    
    jsd_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            P_i = P[i, :].clamp(min=1e-20)
            P_j = P[j, :].clamp(min=1e-20)
            
            M = 0.5 * (P_i + P_j)
            
            log_M = torch.log(M) 
            
            kl_i = F.kl_div(log_M, P_i, reduction='sum', log_target=True)
            kl_j = F.kl_div(log_M, P_j, reduction='sum', log_target=True)
            
            jsd_pair = 0.5 * (kl_i + kl_j)
            jsd_sum += jsd_pair

    return (jsd_sum / pair_count).item() if pair_count > 0 else 0.0

def _calculate_pairwise_kl_div(p1_distributions: torch.Tensor, rollout_n: int) -> float:
    """Calculate the average pairwise symmetric KL divergence between the token distributions of all rollouts"""
    if rollout_n < 2:
        return 0.0
    
    P = p1_distributions.to(torch.float32)

    kl_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            P_i = P[i, :].clamp(min=1e-20)
            P_j = P[j, :].clamp(min=1e-20)
            
            kl_i_j = F.kl_div(P_j.log(), P_i, reduction='sum', log_target=True)
            kl_j_i = F.kl_div(P_i.log(), P_j, reduction='sum', log_target=True)
            
            symmetrized_kl = 0.5 * (kl_i_j + kl_j_i)
            kl_sum += symmetrized_kl

    return (kl_sum / pair_count).item() if pair_count > 0 else 0.0

def _get_response_token_distributions(sequences: torch.Tensor, response_mask: torch.Tensor, rollout_n: int, vocab_size: int) -> torch.Tensor:
    """
    Calculate the vocabulary frequency distribution for n rollouts under the prompt (P1). 
    Return a probability distribution tensor with the shape [rollout_n, vocab_size].
    """
    freqs = torch.zeros(rollout_n, vocab_size, device=sequences.device)
    for j in range(rollout_n):
        valid_tokens = sequences[j, :][response_mask[j, :].bool()]
        if valid_tokens.numel() > 0:
            counts = torch.bincount(valid_tokens, minlength=vocab_size)
            freqs[j] = counts.float()
    
    return F.normalize(freqs, p=1, dim=-1)

def _calculate_distribution_entropy(p: torch.Tensor) -> float:
    """Calculate the Shannon entropy of a single probability distribution p"""
    if p.sum() < 1e-6: 
        return 0.0
    p_clamped = p.clamp(min=1e-20) 
    return (-torch.sum(p_clamped * torch.log(p_clamped))).item()

def _calculate_distribution_confidence(p: torch.Tensor, k: int) -> float:
    """Calculate the Top-k confidence of a single probability distribution p"""
    if p.sum() < 1e-6:
        return 0.0
    
    actual_k = min(k, torch.count_nonzero(p).item())
    if actual_k == 0:
        return 0.0
        
    top_k_probs, _ = torch.topk(p, actual_k)
    top_k_probs_clamped = top_k_probs.clamp(min=1e-20) 
    
    return (-torch.mean(torch.log(top_k_probs_clamped))).item()

def _calculate_pairwise_textual_similarity(responses: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Calculate the average text similarity based on the Levenshtein distance between every pair of rollouts under a prompt.
    Operate directly on the token ID sequence.
    """
    rollout_n = responses.shape[0]
    if rollout_n < 2:
        return 0.0  

    valid_responses = []
    for j in range(rollout_n):
        valid_tokens = responses[j, masks[j, :].bool()].tolist()
        valid_responses.append(valid_tokens)

    similarity_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            seq1 = valid_responses[i]
            seq2 = valid_responses[j]

            distance = Levenshtein.distance(seq1, seq2)
            
            max_len = max(len(seq1), len(seq2))
            if max_len == 0:

                similarity = 1.0
            else:
                similarity = 1.0 - (distance / max_len)
            
            similarity_sum += similarity

    return (similarity_sum / pair_count) if pair_count > 0 else 0.0

def _calculate_pairwise_rouge_l(responses: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Calculate the average text similarity between all pairs of rollouts under a prompt based on the ROUGE-L F1-score.
    """
    rollout_n = responses.shape[0]
    if rollout_n < 2:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    response_strings = []
    for j in range(rollout_n):
        valid_tokens = responses[j, masks[j, :].bool()].tolist()
        response_strings.append(" ".join(map(str, valid_tokens)))

    f1_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            
            scores = scorer.score(response_strings[i], response_strings[j])
            
            f1_sum += scores['rougeL'].fmeasure
            
    return (f1_sum / pair_count) if pair_count > 0 else 0.0

def _get_stacked_token_distribution(sequences: torch.Tensor, response_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Merge all valid tokens from all rollouts under a single prompt and calculate an overall vocabulary word frequency distribution (P1). 
    Return a single probability distribution tensor with a shape of [vocab_size].
    """
    all_valid_tokens = []
    for j in range(sequences.shape[0]):
        valid_tokens = sequences[j, response_mask[j, :].bool()]
        all_valid_tokens.append(valid_tokens)


    if not all_valid_tokens:
        return torch.zeros(vocab_size, device=sequences.device)


    stacked_tokens = torch.cat(all_valid_tokens)
    

    if stacked_tokens.numel() > 0:
        counts = torch.bincount(stacked_tokens, minlength=vocab_size)
        return F.normalize(counts.float(), p=1, dim=-1)
    else:
  
        return torch.zeros(vocab_size, device=sequences.device)


def _calculate_stack_metric(calculator, hidden_states, masks, layer, metric_name):
    valid_rollouts = [hidden_states[j, masks[j, :].bool(), :] for j in range(hidden_states.shape[0])]
    valid_rollouts = [r for r in valid_rollouts if r.shape[0] > 0]
    if not valid_rollouts:
        return np.nan

    stacked_hidden_3d = torch.cat(valid_rollouts, dim=0).unsqueeze(0)
    stacked_mask = torch.ones(1, stacked_hidden_3d.shape[1], device=stacked_hidden_3d.device)

    if "diff" in metric_name:
        stacked_hidden_4d = stacked_hidden_3d.unsqueeze(2)
        results = calculator(hidden_states=stacked_hidden_4d, attention_mask=stacked_mask, compute_diff=True)
        value = results.get(layer, {}).get(metric_name)
    else:
        metric_func = next((func for name, func in calculator.selected_metrics if name == metric_name), None)
        if not metric_func: return np.nan
        value = metric_func(stacked_hidden_3d, stacked_mask)

    return value.item() if value is not None else np.nan

def _calculate_pairwise_rouge_l_on_text(responses: torch.Tensor, masks: torch.Tensor, tokenizer) -> float:
    """
    Calculate the ROUGE-L F1-score between each pair of rollouts under a single prompt based on the decoded text.
    """
    rollout_n = responses.shape[0]
    if rollout_n < 2:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 


    decoded_texts = []
    for j in range(rollout_n):
        valid_token_ids = responses[j, masks[j, :].bool()]

        text = tokenizer.decode(valid_token_ids, skip_special_tokens=True)
        decoded_texts.append(text)


    f1_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            scores = scorer.score(decoded_texts[i], decoded_texts[j])
            f1_sum += scores['rougeL'].fmeasure
            
    return (f1_sum / pair_count) if pair_count > 0 else 0.0

def _calculate_pairwise_rouge_n_s(responses: torch.Tensor, masks: torch.Tensor, rouge_type: str, tokenizer=None) -> float:
    """
    A general pairwise ROUGE calculation function.
    """
    rollout_n = responses.shape[0]
    if rollout_n < 2:
        return 0.0

    use_stemmer = bool(tokenizer)
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)


    sequences_to_score = []
    if tokenizer:

        for j in range(rollout_n):
            valid_token_ids = responses[j, masks[j, :].bool()]
            text = tokenizer.decode(valid_token_ids, skip_special_tokens=True)
            sequences_to_score.append(text)
    else:

        for j in range(rollout_n):
            valid_tokens = responses[j, masks[j, :].bool()].tolist()
            sequences_to_score.append(" ".join(map(str, valid_tokens)))


    f1_sum = 0.0
    pair_count = 0
    for i in range(rollout_n):
        for j in range(i + 1, rollout_n):
            pair_count += 1
            scores = scorer.score(sequences_to_score[i], sequences_to_score[j])
            f1_sum += scores[rouge_type].fmeasure
            
    return (f1_sum / pair_count) if pair_count > 0 else 0.0

def _calculate_n_gram_repetition_rate(token_ids: list, n: int) -> float:
    """
    Calculate the n-gram repetition rate for a sequence of token IDs.
    """
    if len(token_ids) < n:
        return 0.0

    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]

    if not ngrams:
        return 0.0
    ngram_counts = Counter(ngrams)
    
    num_unique_ngrams = len(ngram_counts)
    total_ngrams = len(ngrams)

    repetition_rate = 1.0 - (num_unique_ngrams / total_ngrams)
    
    return repetition_rate
# ==============================================================================
#                 >>>>> 指标注册表 (METRIC_REGISTRY) <<<<<
# ==============================================================================
METRIC_REGISTRY = [
    # --- Rollout Granularity Metrics ---
    {'name': 'Effective Rank', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rollout_calc_results'][j].get('Effective Rank', torch.tensor(np.nan)).item()},
    {'name': 'Effective Rank diff', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rollout_calc_results'][j].get('Effective Rank diff', torch.tensor(np.nan)).item()},
    {'name': 'Effective Rank diff 2', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rollout_calc_results'][j].get('Effective Rank diff 2', torch.tensor(np.nan)).item()},
    {'name': 'test_score_0', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rewards_0'][j]},
    {'name': 'correctness', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['correctness'][j]},
    {'name': 'avg_log_probs', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['avg_log_probs'][j] if data['avg_log_probs'] is not None else np.nan},
    {'name': 'response_entropy', 'granularity': 'rollout', 'calculator_func': lambda data, j: _calculate_distribution_entropy(data['p1_distributions'][j])},
    {'name': 'response_confidence', 'granularity': 'rollout', 'calculator_func': lambda data, j: _calculate_distribution_confidence(data['p1_distributions'][j], data['confidence_k'])},
    {'name': 'n_gram_repetition_rate', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rollout_repetition_rates'][j]},
    {'name': 'response_length', 'granularity': 'rollout', 'calculator_func': lambda data, j: data['rollout_response_lengths'][j]},

  

    # --- Prompt Granularity Metrics ---
    {'name': 'test_score_0_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean(data['rewards_0'])},
    {'name': 'correctness_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean(data['correctness'])},
    {'name': 'Effective Rank Avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean([res.get('Effective Rank', torch.tensor(np.nan)).item() for res in data['rollout_calc_results']])},
    {'name': 'Effective Rank diff Avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean([res.get('Effective Rank diff', torch.tensor(np.nan)).item() for res in data['rollout_calc_results']])},
    {'name': 'Effective Rank diff 2 Avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean([res.get('Effective Rank diff 2', torch.tensor(np.nan)).item() for res in data['rollout_calc_results']])},
    {'name': 'Effective Rank Stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_stack_metric(data['calculator'], data['hidden_states'], data['masks'], data['x_layer'], "Effective Rank")},
    {'name': 'Effective Rank diff Stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_stack_metric(data['calculator'], data['hidden_states'], data['masks'], data['x_layer'], "Effective Rank diff")},
    {'name': 'Effective Rank diff 2 Stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_stack_metric(data['calculator'], data['hidden_states'], data['masks'], data['x_layer'], "Effective Rank diff 2")},
    {'name': 'exploration_token_jsd', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_token_jsd(data['p1_distributions'], data['n'])},
    {'name': 'pairwise_kl_div', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_kl_div(data['p1_distributions'], data['n'])},
    {'name': 'pairwise_euclidean_dist', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_euclidean_dist(data['embeddings'])},
    {'name': 'pairwise_cosine_sim', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_cosine_sim(data['embeddings'])},
    {'name': 'dist_to_center', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_dist_to_center(data['embeddings'])},
    {'name': 'response_entropy_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.mean([_calculate_distribution_entropy(p) for p in data['p1_distributions']])},
    {'name': 'response_confidence_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.mean([_calculate_distribution_confidence(p, data['confidence_k']) for p in data['p1_distributions']])},
    {'name': 'response_entropy_stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_distribution_entropy(data['stacked_p1_distribution'])},
    {'name': 'response_confidence_stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_distribution_confidence(data['stacked_p1_distribution'], data['confidence_k'])},
    # ROUGE-L
    {'name': 'pairwise_rouge_l', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_l(data['responses'], data['masks'])},
    {'name': 'pairwise_rouge_l_text', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_l_on_text(data['responses'], data['masks'], data['tokenizer'])},
    # ROUGE-1
    {'name': 'pairwise_rouge1_token', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_n_s(data['responses'], data['masks'], 'rouge1', tokenizer=None)},
    {'name': 'pairwise_rouge1_text', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_n_s(data['responses'], data['masks'], 'rouge1', tokenizer=data['tokenizer'])},
    # ROUGE-2
    {'name': 'pairwise_rouge2_token', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_n_s(data['responses'], data['masks'], 'rouge2', tokenizer=None)},
    {'name': 'pairwise_rouge2_text', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_pairwise_rouge_n_s(data['responses'], data['masks'], 'rouge2', tokenizer=data['tokenizer'])},
    {'name': 'n_gram_repetition_rate_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.mean(data['rollout_repetition_rates'])},
    {'name': 'n_gram_repetition_rate_stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: _calculate_n_gram_repetition_rate([token for idx in range(data['n']) for token in data['responses'][idx, data['masks'][idx, :].bool()].tolist()], data['repetition_n'])},
    # response len
    {'name': 'response_length_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.mean(data['rollout_response_lengths'])},
    {'name': 'response_length_stack', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.sum(data['rollout_response_lengths'])},
    {'name': 'avg_log_probs_avg', 'granularity': 'prompt', 'calculator_func': lambda data, j: np.nanmean(data['avg_log_probs']) if data['avg_log_probs'] is not None else np.nan},
]
