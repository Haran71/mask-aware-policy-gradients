import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available
from accelerate.utils import gather, gather_object, set_seed
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import print_prompt_completions_sample
import wandb
from types import SimpleNamespace
from .policy_losses import compute_policy_loss_v2
from .position_likelihood import compute_position_logprobs_for_block

if is_peft_available():
    from peft import PeftConfig
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DiffuGRPOTrainer(GRPOTrainer):
    """
    Diffusion policy optimization with StepMerge token and position likelihoods.

    Supports separate token/position losses and the original d1 likelihood fallback.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # Build position loss args for separate position policy loss
        if self.args.use_position_likelihood:
            self._position_loss_args = SimpleNamespace(
                loss_type=self.args.position_loss_type,
                clip_eps=self.args.position_clip_eps,
                clip_eps_low=self.args.position_clip_eps_low,
                clip_eps_high=self.args.position_clip_eps_high,
                averaging_mode=self.args.position_averaging_mode,
            )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids_batched = input_ids.unsqueeze(0)

        # Compute per-token log probs using StepMerge or the original d1 estimator
        pos_logps = None
        if self.args.use_stepmerge:
            unmask_steps = inputs["unmask_steps"]
            # Get sampled block indices for this iteration (None if not using block sampling)
            this_itr_block_indices = None
            if inputs.get("sampled_block_indices") is not None:
                this_itr_block_indices = inputs["sampled_block_indices"][this_itr_idx]
            per_token_logps, pos_logps = self._get_per_token_logps_stepmerge(
                model, input_ids_batched, logits_to_keep, [this_itr_mask_seed],
                unmask_steps, self.args.num_stepmerge_blocks,
                sampled_block_indices=this_itr_block_indices,
            )
            per_token_logps = per_token_logps.squeeze(0)
            if pos_logps is not None:
                pos_logps = pos_logps.squeeze(0)
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids_batched, logits_to_keep, [this_itr_mask_seed])
            per_token_logps = per_token_logps.squeeze(0)

        # Compute KL divergence if needed
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the policy loss via v2 (no thinking/answer split)
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        # Get sample mask for this iteration (None if not using block sampling)
        this_itr_sample_mask = None
        if inputs.get("sample_masks") is not None:
            this_itr_sample_mask = inputs["sample_masks"][this_itr_idx]

        # Get old position logprobs if needed
        old_pos_logps = None
        if self.args.use_position_likelihood and pos_logps is not None:
            if self.num_iterations > 1:
                old_pos_logps = inputs["old_pos_logps"][this_itr_idx].squeeze(0)
            else:
                old_pos_logps = pos_logps.detach()

        pos_policy_loss = torch.tensor(0.0, device=per_token_logps.device)
        pos_loss_stats = {}

        policy_loss, loss_stats, token_clip_mask = compute_policy_loss_v2(
            args=self.args,
            new_logps=per_token_logps,
            old_logps=old_per_token_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            sample_masks=this_itr_sample_mask,
        )

        if self.args.use_position_likelihood and pos_logps is not None:
            pos_policy_loss, pos_loss_stats, _ = compute_policy_loss_v2(
                args=self._position_loss_args,
                new_logps=pos_logps,
                old_logps=old_pos_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                sample_masks=this_itr_sample_mask,
                token_clip_mask=token_clip_mask if self.args.position_mask_token_clipped else None,
            )

        # Total loss
        loss = policy_loss
        if self.args.use_position_likelihood and pos_logps is not None:
            loss = loss + self.args.position_likelihood_lambda * pos_policy_loss
        if self.beta != 0.0:
            kl_mask = completion_mask.float()
            if this_itr_sample_mask is not None:
                kl_mask = kl_mask * this_itr_sample_mask.float()
            kl_loss = (per_token_kl * kl_mask).sum() / kl_mask.sum().clamp(min=1)
            loss = loss + self.beta * kl_loss

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(kl_loss).mean().item())

        self._metrics[mode]["policy/loss"].append(policy_loss.item())

        for key, value in loss_stats.items():
            if isinstance(value, (int, float)):
                self._metrics[mode][f"policy/{key}"].append(value)

        # Advantage stats
        self._metrics[mode]["advantages/mean"].append(advantages.mean().item())
        self._metrics[mode]["advantages/std"].append(advantages.std(correction=int(advantages.numel() > 1)).item())

        if self.args.use_position_likelihood and pos_logps is not None:
            self._metrics[mode]["position/policy_loss"].append(pos_policy_loss.item())
            self._metrics[mode]["position/weighted_loss"].append(
                (self.args.position_likelihood_lambda * pos_policy_loss).item())
            for key, value in pos_loss_stats.items():
                if isinstance(value, (int, float)):
                    self._metrics[mode][f"position/{key}"].append(value)

        if this_itr_sample_mask is not None:
            coverage = (this_itr_sample_mask.float() * completion_mask.float()).sum() / completion_mask.sum()
            self._metrics[mode]["block_sampling/token_coverage"].append(coverage.item())

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        decouple_sampling=False,
        position_sampling_temperature=0.0,
        stochastic_position_selection=False,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)

        Args:
            decouple_sampling: When True, position selection (which tokens to unmask)
                and token selection (what token to place) use independent noise sources.
                Position ordering uses clean max-prob confidence (optionally with
                position_sampling_temperature noise). Token choice uses Gumbel noise.
                When False, uses the original coupled approach.
            position_sampling_temperature: Temperature for stochastic position selection
                (only used when decouple_sampling=True). Adds Gumbel noise to
                log(max_prob) for position ranking. 0.0 = deterministic ordering.
        """
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            # Track when each completion token was unmasked (-1 = still masked)
            unmask_steps = torch.full((bs, gen_length), -1, dtype=torch.long, device=model.device)
            step_counter = 0

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for token sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking / position confidence
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)

                                if decouple_sampling:
                                    # Decoupled: position confidence = max prob (independent of token noise)
                                    x0_p = p.max(dim=-1).values

                                    # Optionally add Gumbel noise for stochastic position selection
                                    if position_sampling_temperature > 0.0:
                                        log_conf = torch.log(x0_p + 1e-8)
                                        pos_noise = torch.rand_like(log_conf, dtype=dtype)
                                        pos_gumbel = -torch.log(-torch.log(pos_noise + 1e-8) + 1e-8)
                                        x0_p = log_conf + pos_gumbel * position_sampling_temperature
                                else:
                                    # Original coupled: confidence = prob of the Gumbel-sampled token
                                    x0_p = torch.squeeze(
                                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                    )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Optionally add Gumbel noise for stochastic position selection
                            if stochastic_position_selection and position_sampling_temperature > 0.0:
                                finite_mask = confidence > -np.inf
                                noise = torch.rand_like(confidence, dtype=dtype)
                                gumbel = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
                                confidence = torch.where(
                                    finite_mask,
                                    confidence + gumbel * position_sampling_temperature,
                                    confidence,
                                )

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]

                            # Record unmask step for completion tokens that were just transferred
                            completion_transfer = transfer_index[:, prompt.shape[1]:]
                            unmask_steps[completion_transfer] = step_counter

                            step_counter += 1
                            del x0, confidence, transfer_index

            return x, unmask_steps

    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        logits = model(batch).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _get_per_token_logps_stepmerge(self, model, input_ids, logits_to_keep, mask_seeds, unmask_steps, N, sampled_block_indices=None):
        """
        StepMerge log probability estimator.

        Partitions the diffusion trajectory into N blocks based on unmask_steps.
        For each block, reconstructs the masked state at the block start boundary,
        runs a forward pass, and computes log probs for tokens that transitioned
        (were unmasked) in that block.

        Args:
            model: Policy model
            input_ids: [num_iterations, batch_size, seq_len]
            logits_to_keep: int - completion length
            mask_seeds: list[int] - unused, kept for API consistency
            unmask_steps: [batch_size, logits_to_keep] - step each token was unmasked
            N: int - number of blocks to partition trajectory into
            sampled_block_indices: list[int] or None - if provided, only process these
                block indices (for K-block sampling). When None, processes all N blocks.

        Returns:
            per_token_logps: [num_iterations, batch_size, logits_to_keep]
            per_token_pos_logps: [num_iterations, batch_size, logits_to_keep] or None
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device

        T = self.args.diffusion_steps
        block_size = T // N
        prompt_length = seq_len - logits_to_keep

        # Use first iteration's input (all iterations are identical for StepMerge)
        clean_sequences = input_ids[0]  # [batch_size, seq_len]

        # Iterate over sampled blocks (K-block sampling) or all N blocks
        block_indices = sampled_block_indices if sampled_block_indices is not None else list(range(N))
        blocks_to_process = []
        for n in block_indices:
            t_start = n * block_size
            t_end = (n + 1) * block_size
            # Check if any token in the batch transitioned in this block
            mask_start = (unmask_steps > t_start) | (unmask_steps == -1)
            mask_end = (unmask_steps > t_end) | (unmask_steps == -1)
            transitioned = mask_start & ~mask_end  # [batch_size, logits_to_keep]
            if transitioned.any():
                blocks_to_process.append((n, t_start, t_end))

        # Accumulators: average when multiple blocks contribute to the same position
        batch_logps = torch.zeros(batch_size, logits_to_keep, device=device, dtype=torch.float32)
        logps_count = torch.zeros(batch_size, logits_to_keep, device=device, dtype=torch.int32)

        # Position log-prob accumulators
        compute_pos = self.args.use_position_likelihood
        if compute_pos:
            batch_pos_logps = torch.zeros(batch_size, logits_to_keep, device=device, dtype=torch.float32)
            pos_logps_count = torch.zeros(batch_size, logits_to_keep, device=device, dtype=torch.int32)

        microbatch_size = self.args.stepmerge_blocks_per_microbatch

        # Build (block, sequence) entries — all sequences active per block for DDP correctness
        active_entries = []
        for _, t_start, t_end in blocks_to_process:
            mask_start = (unmask_steps > t_start) | (unmask_steps == -1)
            mask_end = (unmask_steps > t_end) | (unmask_steps == -1)
            transitioned = mask_start & ~mask_end  # [batch_size, logits_to_keep]

            for seq_idx in range(batch_size):
                active_entries.append({
                    'seq_idx': seq_idx,
                    't_start': t_start,
                    't_end': t_end,
                    'transition_mask': transitioned[seq_idx].clone(),  # [logits_to_keep]
                })

        # Process entries in micro-batches
        prompt_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_mask[:prompt_length] = True

        for mb_start in range(0, len(active_entries), microbatch_size):
            mb_entries = active_entries[mb_start:mb_start + microbatch_size]
            mb_size = len(mb_entries)
            if mb_size == 0:
                continue

            # Build masked inputs for this micro-batch
            x_t_start_list = []
            for entry in mb_entries:
                seq_idx = entry['seq_idx']
                t_start = entry['t_start']

                single_seq = clean_sequences[seq_idx:seq_idx+1].clone()  # [1, seq_len]

                # Mask completion tokens that were unmasked after t_start (or never unmasked)
                single_unmask = unmask_steps[seq_idx:seq_idx+1]  # [1, logits_to_keep]
                should_mask = (single_unmask > t_start) | (single_unmask == -1)
                single_seq[0, prompt_length:][should_mask.squeeze(0)] = self.args.mask_id

                x_t_start_list.append(single_seq)

            x_t_start_batch = torch.cat(x_t_start_list, dim=0)  # [mb_size, seq_len]

            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.get_logits(
                    model, x_t_start_batch, prompt_mask,
                    self.args.cfg_scale, self.args.mask_id
                )

            # Extract completion logits
            logits = logits[:, -logits_to_keep:, :]  # [mb_size, logits_to_keep, vocab]

            # Clean targets for cross-entropy
            clean_targets = torch.stack(
                [clean_sequences[e['seq_idx'], -logits_to_keep:] for e in mb_entries], dim=0
            )

            # Stack transition masks
            all_transitions = torch.stack([e['transition_mask'] for e in mb_entries], dim=0)

            # Sparse log prob computation on transitioned tokens only
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = clean_targets.reshape(-1)
            transitions_flat = all_transitions.reshape(-1)

            transition_indices = transitions_flat.nonzero(as_tuple=True)[0]

            if len(transition_indices) > 0:
                sparse_logits = logits_flat[transition_indices]
                sparse_targets = targets_flat[transition_indices]

                sparse_log_probs = F.log_softmax(sparse_logits.float(), dim=-1)
                sparse_token_logps = sparse_log_probs.gather(1, sparse_targets.unsqueeze(1)).squeeze(1)

                # Map back to full tensor
                logps_full = torch.zeros_like(transitions_flat, dtype=torch.float32)
                logps_full[transition_indices] = sparse_token_logps
                logps_reshaped = logps_full.view(mb_size, logits_to_keep)

                # Accumulate into batch results
                for i, entry in enumerate(mb_entries):
                    seq_idx = entry['seq_idx']
                    tmask = entry['transition_mask']
                    batch_logps[seq_idx][tmask] += logps_reshaped[i][tmask]
                    logps_count[seq_idx][tmask] += 1

                    # Position log-probs from same forward pass logits
                    if compute_pos:
                        pos_lp = compute_position_logprobs_for_block(
                            logits=logits[i],
                            transition_mask=tmask,
                            unmask_steps=unmask_steps[seq_idx],
                            target_ids=clean_targets[i],
                            t_start=entry['t_start'],
                            t_end=entry['t_end'],
                            method=self.args.position_likelihood_method,
                            competition_scope=self.args.position_likelihood_scope,
                            confidence_type=self.args.position_likelihood_confidence,
                            tau=self.args.position_likelihood_temperature,
                            confidence_source=self.args.position_confidence_source,
                        )
                        batch_pos_logps[seq_idx][tmask] += pos_lp[tmask]
                        pos_logps_count[seq_idx][tmask] += 1

            del logits, x_t_start_batch, x_t_start_list

        # Average log probs at positions covered by multiple blocks
        safe_count = logps_count.clamp(min=1)
        batch_logps = batch_logps / safe_count.float()

        per_token_pos_logps = None
        if compute_pos:
            safe_pos_count = pos_logps_count.clamp(min=1)
            batch_pos_logps = batch_pos_logps / safe_pos_count.float()
            per_token_pos_logps = batch_pos_logps.unsqueeze(0).repeat(num_iterations, 1, 1)

        # Repeat for all iterations (deterministic — no need to recompute)
        per_token_logps = batch_logps.unsqueeze(0).repeat(num_iterations, 1, 1)

        return per_token_logps, per_token_pos_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self._step == 0 and self.state.global_step > 0:
                self._step = self.state.global_step * self.args.gradient_accumulation_steps
            buffer_index = self._step % self.args.gradient_accumulation_steps
            # Rollout buffers are not checkpointed; refill each slot after resume.
            if self.state.global_step % self.num_iterations == 0 or self._buffered_inputs[buffer_index] is None:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = self._buffered_inputs[buffer_index]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            unmask_steps_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_completion_ids, batch_unmask_steps = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                    decouple_sampling=self.args.decouple_position_token_sampling,
                    position_sampling_temperature=self.args.position_sampling_temperature,
                    stochastic_position_selection=self.args.stochastic_position_selection,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                unmask_steps_all.append(batch_unmask_steps)

                del batch_prompt_ids, batch_prompt_completion_ids, batch_unmask_steps
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
            unmask_steps = torch.cat(unmask_steps_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        all_old_pos_logps = None
        with torch.no_grad():
            if self.num_iterations > 1:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                if self.args.use_stepmerge:
                    old_per_token_logps, old_pos_logps = self._get_per_token_logps_stepmerge(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds,
                        unmask_steps, self.args.num_stepmerge_blocks,
                    )
                    all_old_pos_logps = old_pos_logps
                else:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                    )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    if self.args.use_stepmerge:
                        ref_per_token_logps, _ = self._get_per_token_logps_stepmerge(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds,
                            unmask_steps, self.args.num_stepmerge_blocks,
                        )
                    else:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                        )
                    all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = getattr(reward_func, '__name__', None) or reward_func.func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute advantages based on estimator type
        rewards_grouped = rewards.view(-1, self.num_generations)

        if self.args.advantage_estimator == "rloo":
            sum_grouped_rewards = rewards_grouped.sum(dim=1, keepdim=True)
            mean_others = (sum_grouped_rewards - rewards_grouped) / (self.num_generations - 1)
            advantages_grouped = rewards_grouped - mean_others
        elif self.args.advantage_estimator == "grpo":
            mean_grouped_rewards = rewards_grouped.mean(dim=1, keepdim=True)
            advantages_grouped = rewards_grouped - mean_grouped_rewards
        elif self.args.advantage_estimator == "none":
            advantages_grouped = rewards_grouped
        else:
            raise ValueError(f"Unknown advantage_estimator: {self.args.advantage_estimator}")

        advantages = advantages_grouped.view(-1)
        std_grouped_rewards = rewards_grouped.std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = getattr(reward_func, '__name__', None) or reward_func.func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # Sample StepMerge blocks for each policy update
        all_sampled_block_indices = None
        all_sample_masks = None
        K = self.args.stepmerge_sample_k_blocks
        if self.args.use_stepmerge and K > 0:
            N = self.args.num_stepmerge_blocks
            T = self.args.diffusion_steps
            block_size = T // N

            # Filter to blocks that have at least one valid token (transitioned AND before EOS)
            # in ANY sequence in the batch
            completion_mask_bool = completion_mask.bool()
            valid_block_indices = []
            for n in range(N):
                t_start = n * block_size
                t_end = (n + 1) * block_size
                mask_start = (unmask_steps > t_start) | (unmask_steps == -1)
                mask_end = (unmask_steps > t_end) | (unmask_steps == -1)
                transitioned = mask_start & ~mask_end
                if (transitioned & completion_mask_bool).any():
                    valid_block_indices.append(n)
            V = len(valid_block_indices)
            valid_tensor = torch.tensor(valid_block_indices, dtype=torch.long) if V > 0 else None

            all_sampled_block_indices = []
            sample_mask_list = []
            for iter_idx in range(self.num_iterations):
                # Deterministic seed per (step, iteration) for reproducibility
                rng = torch.Generator()
                rng.manual_seed(self.state.global_step * self.num_iterations + iter_idx)

                if V == 0:
                    # No valid blocks: fall back to original behavior
                    perm = torch.randperm(N, generator=rng)
                    sampled = perm[:K].sort().values.tolist()
                elif V <= K:
                    # Fewer valid blocks than K: use all valid blocks
                    _ = torch.randperm(V, generator=rng)  # consume rng for state consistency
                    sampled = sorted(valid_block_indices)
                else:
                    # Sample K from valid blocks
                    perm = torch.randperm(V, generator=rng)
                    sampled = valid_tensor[perm[:K]].sort().values.tolist()

                all_sampled_block_indices.append(sampled)

                # Compute sample_mask: which tokens are covered by sampled blocks
                sample_mask = torch.zeros_like(unmask_steps, dtype=torch.bool)
                for block_n in sampled:
                    t_start = block_n * block_size
                    t_end = (block_n + 1) * block_size
                    mask_start = (unmask_steps > t_start) | (unmask_steps == -1)
                    mask_end = (unmask_steps > t_end) | (unmask_steps == -1)
                    transitioned = mask_start & ~mask_end
                    sample_mask |= transitioned
                sample_mask_list.append(sample_mask)

            all_sample_masks = torch.stack(sample_mask_list, dim=0)  # [num_iterations, batch, logits_to_keep]

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
            "unmask_steps": unmask_steps,
            "old_pos_logps": all_old_pos_logps,
            "sampled_block_indices": all_sampled_block_indices,
            "sample_masks": all_sample_masks,
        }

# Modification notice: Adapted for Mask-Aware Policy Gradients.
