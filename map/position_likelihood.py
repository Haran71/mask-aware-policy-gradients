"""
Position-Aware Likelihood for StepMerge.

Computes per-token position log-probability capturing "was this position
selected at the right time?" based on model confidence and the actual
generation ordering from unmask_steps.
"""

import torch
import torch.nn.functional as F


def compute_confidence(logits: torch.Tensor, method: str = "max_logit") -> torch.Tensor:
    """
    Compute per-position confidence from logits.

    Args:
        logits: [N, vocab_size]
        method: 'max_logit', 'top_prob', 'neg_entropy', or 'margin'

    Returns:
        [N] confidence scores
    """
    if method == "max_logit":
        return logits.max(dim=-1).values
    elif method == "top_prob":
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values  # highest softmax probability
    elif method == "neg_entropy":
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return (probs * log_probs).sum(dim=-1)  # negative entropy (higher = more certain)
    elif method == "margin":
        probs = F.softmax(logits, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def compute_position_logprobs_bradley_terry(
    conf_trans: torch.Tensor,
    all_conf: torch.Tensor,
    competitor_mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Bradley-Terry position log-probability.

    Args:
        conf_trans: [num_trans] confidence of transitioning tokens
        all_conf: [L] confidence of all completion positions
        competitor_mask: [num_trans, L] True where position j is a competitor of token i
        tau: temperature

    Returns:
        [num_trans] position log-probabilities
    """
    diff = (conf_trans.unsqueeze(1) - all_conf.unsqueeze(0)) / tau  # [num_trans, L]
    log_sigma = F.logsigmoid(diff)  # [num_trans, L]
    return (log_sigma * competitor_mask.float()).sum(dim=-1)  # [num_trans]


def compute_position_logprobs_softmax(
    conf_trans: torch.Tensor,
    all_conf: torch.Tensor,
    competitor_mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Softmax position log-probability.

    Args:
        conf_trans: [num_trans] confidence of transitioning tokens
        all_conf: [L] confidence of all completion positions
        competitor_mask: [num_trans, L] True where position j is a competitor of token i
        tau: temperature

    Returns:
        [num_trans] position log-probabilities
    """
    self_score = conf_trans.unsqueeze(1) / tau  # [num_trans, 1]
    comp_scores = (all_conf.unsqueeze(0) / tau).expand(conf_trans.size(0), -1)  # [num_trans, L]
    comp_scores = comp_scores.masked_fill(~competitor_mask, float('-inf'))
    all_scores = torch.cat([self_score, comp_scores], dim=1)  # [num_trans, 1 + L]
    return (conf_trans / tau) - torch.logsumexp(all_scores, dim=1)  # [num_trans]


def compute_position_logprobs_for_block(
    logits: torch.Tensor,
    transition_mask: torch.Tensor,
    unmask_steps: torch.Tensor,
    target_ids: torch.Tensor,
    t_start: int,
    t_end: int,
    method: str = "bradley_terry",
    competition_scope: str = "position",
    confidence_type: str = "max_logit",
    tau: float = 1.0,
    confidence_source: str = "gt",
) -> torch.Tensor:
    """
    Compute position log-probs for all transitioning tokens in one (block, sequence) pair.

    For transitioning tokens, confidence depends on confidence_source:
      - 'gt': logit of the ground truth token
      - 'max': highest logit (same measure as competitors)
    For all competitors, confidence is the max logit (or other measure).

    Args:
        logits: [logits_to_keep, vocab_size] from the block's forward pass
        transition_mask: [logits_to_keep] True for tokens transitioning in this block
        unmask_steps: [logits_to_keep] diffusion step each token was unmasked at (-1 = never)
        target_ids: [logits_to_keep] ground truth token IDs for each position
        t_start: block start step
        t_end: block end step
        method: 'bradley_terry' or 'softmax'
        competition_scope: 'segment' (future segments only) or 'position' (+ within-segment by unmask order)
        confidence_type: 'max_logit', 'neg_entropy', or 'margin'
        tau: temperature
        confidence_source: 'gt' (ground truth logit) or 'max' (highest logit)

    Returns:
        [logits_to_keep] position log-probs (0 for non-transitioning tokens)
    """
    device = logits.device
    L = logits.size(0)
    result = torch.zeros(L, device=device, dtype=logits.dtype)

    if not transition_mask.any():
        return result

    # Max-logit confidence for all positions (used for competitors)
    all_conf = compute_confidence(logits, confidence_type)  # [L]

    # Transitioning token indices
    trans_indices = transition_mask.nonzero(as_tuple=True)[0]  # [num_trans]

    # Confidence for transitioning tokens
    if confidence_source == "max":
        # Use the same confidence measure as competitors (highest logit / neg_entropy / margin / top_prob)
        conf_trans = all_conf[trans_indices]  # [num_trans]
    else:
        # Default 'gt': use ground truth token's value as confidence
        if confidence_type == "max_logit":
            # GT token's raw logit
            conf_trans = logits[trans_indices].gather(1, target_ids[trans_indices].unsqueeze(1)).squeeze(1)
        elif confidence_type == "top_prob":
            # GT token's softmax probability
            probs = F.softmax(logits[trans_indices], dim=-1)
            conf_trans = probs.gather(1, target_ids[trans_indices].unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(
                f"confidence_source='gt' is only compatible with confidence_type='max_logit' or 'top_prob', "
                f"got confidence_type='{confidence_type}'. Use confidence_source='max' with "
                f"'{confidence_type}' so both transitioning tokens and competitors are on the same scale."
            )

    # Build competitor mask: [num_trans, L]
    # Future segments: unmasked after this block
    future = (unmask_steps > t_end) & (unmask_steps != -1)  # [L]
    competitor_matrix = future.unsqueeze(0).expand(len(trans_indices), -1).clone()  # [num_trans, L]

    # Position-level: also compete with same-segment tokens unmasked later
    if competition_scope == "position":
        trans_steps = unmask_steps[trans_indices]  # [num_trans]
        same_seg = (unmask_steps > t_start) & (unmask_steps <= t_end) & (unmask_steps != -1)  # [L]
        same_seg_steps = torch.where(same_seg, unmask_steps, torch.tensor(-1, device=device, dtype=unmask_steps.dtype))
        competitor_matrix |= (same_seg_steps.unsqueeze(0) > trans_steps.unsqueeze(1))

    # Exclude self
    competitor_matrix[torch.arange(len(trans_indices), device=device), trans_indices] = False

    # Compute position log-probs
    if method == "bradley_terry":
        pos_logprobs = compute_position_logprobs_bradley_terry(conf_trans, all_conf, competitor_matrix, tau)
    elif method == "softmax":
        pos_logprobs = compute_position_logprobs_softmax(conf_trans, all_conf, competitor_matrix, tau)
    else:
        raise ValueError(f"Unknown position likelihood method: {method}")

    result[trans_indices] = pos_logprobs.to(result.dtype)
    return result

# Modification notice: Adapted for Mask-Aware Policy Gradients.
