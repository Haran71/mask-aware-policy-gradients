"""
Policy Loss Functions V2 for D1

Simplified version — no thinking/answer token split.
All completion tokens are treated uniformly with the same advantages.

Supports: REINFORCE, GRPO, DAPO, GSPO, GSPO_TOKEN
Averaging: token_level, sequence_level
"""

import torch
from typing import Dict, Tuple


def compute_policy_loss_v2(
    args,
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    sample_masks: torch.Tensor = None,
    token_clip_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    """
    Compute policy gradient loss over all completion tokens uniformly.

    Args:
        args: Config with loss_type, averaging_mode, clip_eps, clip_eps_low, clip_eps_high
        new_logps: Current policy log probabilities [batch, seq_len]
        old_logps: Old policy log probabilities [batch, seq_len]
        advantages: Advantage estimates [batch] or [batch, 1]
        completion_mask: Mask for valid completion tokens [batch, seq_len]
        sample_masks: Optional mask for partial log probs [batch, seq_len]
        token_clip_mask: Optional bool tensor [batch, seq_len]. True = token was clipped
            in another loss (e.g. token policy loss). When provided, clipped tokens are masked
            out and remaining tokens use unclipped importance ratios.

    Returns:
        loss: Policy gradient loss (scalar)
        stats: Dictionary of statistics
        clip_mask: Bool tensor [batch, seq_len] where ratio was clipped, or None
    """
    stats = {}
    clip_mask = None

    if advantages.dim() > 1:
        advantages = advantages.squeeze(-1)

    token_mask = completion_mask.clone().float()
    if sample_masks is not None:
        token_mask = token_mask * sample_masks.float()
        stats["sampled_tokens_ratio"] = sample_masks.float().mean().item()

    if args.loss_type == "reinforce":
        token_losses = _reinforce(new_logps, advantages, token_mask)
        stats["loss_type"] = "reinforce"

    elif args.loss_type == "grpo":
        token_losses, loss_stats, clip_mask = _grpo(
            new_logps, old_logps, advantages, token_mask, args.clip_eps,
            token_clip_mask=token_clip_mask,
        )
        stats.update(loss_stats)
        stats["loss_type"] = "grpo"

    elif args.loss_type == "dapo":
        token_losses, loss_stats, clip_mask = _dapo(
            new_logps, old_logps, advantages, token_mask,
            args.clip_eps_low, args.clip_eps_high,
            token_clip_mask=token_clip_mask,
        )
        stats.update(loss_stats)
        stats["loss_type"] = "dapo"

    elif args.loss_type == "gspo":
        if sample_masks is not None:
            pg_loss, loss_stats = _gspo_partial(
                new_logps, old_logps, advantages, token_mask, completion_mask,
                sample_masks, args.clip_eps_low, args.clip_eps_high,
            )
        else:
            pg_loss, loss_stats = _gspo(
                new_logps, old_logps, advantages, token_mask, completion_mask,
                args.clip_eps_low, args.clip_eps_high,
            )
        stats.update(loss_stats)
        stats["loss_type"] = "gspo"
        return pg_loss, stats, None

    elif args.loss_type == "gspo_token":
        token_losses, loss_stats = _gspo_token(
            new_logps, old_logps, advantages, token_mask, completion_mask,
            args.clip_eps_low, args.clip_eps_high,
        )
        stats.update(loss_stats)
        stats["loss_type"] = "gspo_token"

    else:
        raise ValueError(f"Invalid loss_type: {args.loss_type}")

    # Averaging
    if args.averaging_mode == "sequence_level":
        pg_loss = token_losses.sum(dim=1)
        valid_mask = (completion_mask.sum(dim=1) > 0).float()
        pg_loss = (pg_loss * valid_mask).mean()
        stats["averaging"] = "sequence_level"

    elif args.averaging_mode == "token_level":
        total_valid_tokens = token_mask.sum()
        if total_valid_tokens > 0:
            pg_loss = (token_losses * completion_mask.float()).sum() / total_valid_tokens
        else:
            # token_losses is already all zeros (token_mask was 0) but stays
            # graph-connected to the model. Using .sum() instead of a detached
            # leaf tensor ensures backward() traverses the model and fires
            # DeepSpeed's gradient hooks on all ranks uniformly.
            pg_loss = token_losses.sum()
        stats["averaging"] = "token_level"
        stats["total_tokens"] = total_valid_tokens.item()

    else:
        raise ValueError(f"Invalid averaging_mode: {args.averaging_mode}")

    return pg_loss, stats, clip_mask


# ---------------------------------------------------------------------------
# Per-token loss functions
# ---------------------------------------------------------------------------

def _reinforce(
    new_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    return -advantages.detach() * new_logps * token_mask


def _grpo(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    clip_eps: float,
    token_clip_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    ratio = torch.exp(new_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    clip_mask = (ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)

    if token_clip_mask is not None:
        # Unclipped importance-weighted loss, with externally-clipped tokens masked out
        keep_mask = (~token_clip_mask).float()
        token_losses = -ratio * advantages.detach() * token_mask * keep_mask
    else:
        surr1 = ratio * advantages.detach() * token_mask
        surr2 = clipped_ratio * advantages.detach() * token_mask
        token_losses = -torch.min(surr1, surr2)

    # Compute stats only over valid (unmasked) positions
    valid = token_mask.bool()
    valid_ratio = ratio[valid] if valid.any() else ratio
    stats = {
        "clipfrac": clip_mask[valid].float().mean().item() if valid.any() else 0.0,
        "ratio_mean": valid_ratio.mean().item(),
        "ratio_max": valid_ratio.max().item(),
        "ratio_min": valid_ratio.min().item(),
    }
    if token_clip_mask is not None:
        stats["token_clip_masked_frac"] = token_clip_mask.float().mean().item()
    return token_losses, stats, clip_mask


def _dapo(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
    token_clip_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    ratio = torch.exp(new_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps_low, 1 + clip_eps_high)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    clip_mask = (ratio < 1 - clip_eps_low) | (ratio > 1 + clip_eps_high)

    if token_clip_mask is not None:
        # Unclipped importance-weighted loss, with externally-clipped tokens masked out
        keep_mask = (~token_clip_mask).float()
        token_losses = -ratio * advantages.detach() * token_mask * keep_mask
    else:
        surr1 = ratio * advantages.detach() * token_mask
        surr2 = clipped_ratio * advantages.detach() * token_mask
        token_losses = -torch.min(surr1, surr2)

    # Compute stats only over valid (unmasked) positions
    valid = token_mask.bool()
    valid_ratio = ratio[valid] if valid.any() else ratio
    clipfrac_low = (valid_ratio < 1 - clip_eps_low).float().mean()
    clipfrac_high = (valid_ratio > 1 + clip_eps_high).float().mean()
    stats = {
        "clipfrac_low": clipfrac_low.item(),
        "clipfrac_high": clipfrac_high.item(),
        "clipfrac_total": (clipfrac_low + clipfrac_high).item(),
        "ratio_mean": valid_ratio.mean().item(),
        "ratio_max": valid_ratio.max().item(),
        "ratio_min": valid_ratio.min().item(),
    }
    if token_clip_mask is not None:
        stats["token_clip_masked_frac"] = token_clip_mask.float().mean().item()
    return token_losses, stats, clip_mask


# ---------------------------------------------------------------------------
# Sequence-level losses (GSPO)
# ---------------------------------------------------------------------------

def _gspo(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
) -> Tuple[torch.Tensor, Dict]:
    if advantages.dim() > 1:
        advantages = advantages.squeeze(-1)

    logp_diff = (new_logps - old_logps) * token_mask
    seq_lengths = torch.clamp(token_mask.sum(dim=1), min=1.0)
    avg_logp_diff = logp_diff.sum(dim=1) / seq_lengths

    importance_ratios = torch.exp(avg_logp_diff)
    clipped_ratios = torch.clamp(importance_ratios, 1 - clip_eps_low, 1 + clip_eps_high)

    policy_loss = torch.min(
        importance_ratios * advantages.detach(),
        clipped_ratios * advantages.detach(),
    )

    valid_mask = (completion_mask.sum(dim=1) > 0).float()
    valid_count = valid_mask.sum()
    if valid_count > 0:
        loss = -(policy_loss * valid_mask).sum() / valid_count
    else:
        loss = torch.tensor(0.0, device=new_logps.device, requires_grad=True)

    stats = {
        "gspo/importance_ratio_mean": importance_ratios.mean().item(),
        "gspo/importance_ratio_std": importance_ratios.std(correction=int(importance_ratios.numel() > 1)).item(),
        "gspo/importance_ratio_max": importance_ratios.max().item(),
        "gspo/importance_ratio_min": importance_ratios.min().item(),
        "gspo/clipped_fraction": (importance_ratios != clipped_ratios).float().mean().item(),
        "gspo/avg_sequence_length": seq_lengths.float().mean().item(),
        "gspo/valid_sequences": valid_count.item(),
        "averaging": "sequence_level_internal",
    }
    return loss, stats


def _gspo_partial(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    sample_masks: torch.Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
) -> Tuple[torch.Tensor, Dict]:
    if advantages.dim() > 1:
        advantages = advantages.squeeze(-1)

    combined_mask = token_mask * sample_masks.float()
    logp_diff = (new_logps - old_logps) * combined_mask
    sampled_seq_lengths = torch.clamp(combined_mask.sum(dim=1), min=1.0)
    avg_logp_diff = logp_diff.sum(dim=1) / sampled_seq_lengths

    importance_ratios = torch.exp(avg_logp_diff)
    clipped_ratios = torch.clamp(importance_ratios, 1 - clip_eps_low, 1 + clip_eps_high)

    policy_loss = torch.min(
        importance_ratios * advantages.detach(),
        clipped_ratios * advantages.detach(),
    )

    valid_mask = (completion_mask.sum(dim=1) > 0).float()
    valid_count = valid_mask.sum()
    if valid_count > 0:
        loss = -(policy_loss * valid_mask).sum() / valid_count
    else:
        loss = torch.tensor(0.0, device=new_logps.device, requires_grad=True)

    total_tokens = torch.clamp(token_mask.sum(dim=1), min=1.0)
    sampling_rate = (sampled_seq_lengths / total_tokens).mean()

    stats = {
        "gspo_partial/importance_ratio_mean": importance_ratios.mean().item(),
        "gspo_partial/importance_ratio_std": importance_ratios.std(correction=int(importance_ratios.numel() > 1)).item(),
        "gspo_partial/clipped_fraction": (importance_ratios != clipped_ratios).float().mean().item(),
        "gspo_partial/avg_sampled_length": sampled_seq_lengths.float().mean().item(),
        "gspo_partial/sampling_rate": sampling_rate.item(),
        "gspo_partial/valid_sequences": valid_count.item(),
        "averaging": "sequence_level_partial",
    }
    return loss, stats


def _gspo_token(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
) -> Tuple[torch.Tensor, Dict]:
    if advantages.dim() > 1:
        advantages = advantages.squeeze(-1)

    logp_diff = (new_logps - old_logps) * token_mask
    seq_lengths = torch.clamp(token_mask.sum(dim=1), min=1.0)
    avg_logp_diff = logp_diff.sum(dim=1) / seq_lengths

    # Sequence-level importance ratio (NOT clipped)
    seq_ratios = torch.exp(avg_logp_diff)

    # Token-level: si,t = sg[si] * pi_theta(t) / sg[pi_theta(t)]
    current_token_probs = torch.exp(new_logps)
    token_ratios = seq_ratios.detach().unsqueeze(1) * (current_token_probs / current_token_probs.detach())

    clipped_token_ratios = torch.clamp(token_ratios, 1 - clip_eps_low, 1 + clip_eps_high)
    token_advantages = advantages.unsqueeze(1).expand_as(token_mask)

    surr1 = token_ratios * token_advantages.detach() * token_mask
    surr2 = clipped_token_ratios * token_advantages.detach() * token_mask
    token_losses = -torch.min(surr1, surr2)

    stats = {
        "gspo_token/seq_ratio_mean": seq_ratios.mean().item(),
        "gspo_token/token_ratio_mean": token_ratios.mean().item(),
        "gspo_token/clipped_fraction": (token_ratios != clipped_token_ratios).float().mean().item(),
        "gspo_token/avg_sequence_length": seq_lengths.float().mean().item(),
    }
    return token_losses, stats

# Modification notice: Adapted for Mask-Aware Policy Gradients.
