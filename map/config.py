# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional
from trl import GRPOConfig


@dataclass
class DiffuGRPOConfig(GRPOConfig):
    """GSM8K/MATH diffusion trainer settings; standard options come from TRL."""

    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    model_path: Optional[str] = field(
        default="",
    )
    dora: bool = field(
        default=False,
        metadata={
            "help": "Use DoRA (Weight-Decomposed Low-Rank Adaptation) instead of standard LoRA."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    advantage_estimator: str = field(
        default="rloo",
        metadata={
            "help": "Advantage estimation method. Options: 'rloo' (REINFORCE Leave-One-Out, default), 'grpo' "
            "(Group Relative Policy Optimization), or 'none' (use rewards directly as advantages). "
            "RLOO computes advantages relative to mean of other samples (lower variance), "
            "GRPO uses mean of all samples including current (simpler baseline), "
            "'none' uses rewards without a baseline."
        },
    )
    generation_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."
        },
    )
    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    diffusion_steps: Optional[int] = field(
        default=64,
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    decouple_position_token_sampling: bool = field(
        default=False,
        metadata={
            "help": "Decouple position and token sampling during generation. "
            "When True, position selection uses clean max-prob confidence "
            "(deterministic or with position_sampling_temperature noise), "
            "and token selection uses Gumbel noise independently. "
            "When False (default), uses the original coupled approach where "
            "Gumbel noise on token logits also affects position ordering."
        },
    )
    position_sampling_temperature: float = field(
        default=0.0,
        metadata={
            "help": "Gumbel noise temperature for either position sampling mode. "
            "Decoupled sampling perturbs log(max_prob); stochastic selection perturbs "
            "the selected remasking confidence. Zero disables explicit position noise."
        },
    )
    stochastic_position_selection: bool = field(
        default=False,
        metadata={
            "help": "When True, position selection adds Gumbel noise (scaled by "
            "position_sampling_temperature) to confidence scores before top-k, "
            "making position selection stochastic. Works with any remasking strategy. "
            "Unlike decouple_position_token_sampling, this does not change the confidence "
            "measure itself, only adds noise before the top-k selection step."
        },
    )
    p_mask_prompt: float = field(
        default=0.3,
        metadata={"help": "Probability of masking the prompt."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    random_masking: bool = field(
        default=True,
        metadata={"help": "Whether to randomly mask tokens."},
    )
    compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for model and training: 'bfloat16', 'float16', or 'float32'."},
    )
    use_stepmerge: bool = field(
        default=False,
        metadata={"help": "Use d2-StepMerge log probability estimator (partition trajectory into N blocks)"},
    )
    num_stepmerge_blocks: int = field(
        default=8,
        metadata={"help": "Number of blocks N to partition trajectory into for StepMerge (paper uses 16-32)"},
    )
    stepmerge_blocks_per_microbatch: int = field(
        default=1,
        metadata={"help": "Number of StepMerge blocks to process in each micro-batch for memory efficiency. Lower = less memory, slower. Set to num_stepmerge_blocks to process all at once."},
    )
    stepmerge_sample_k_blocks: int = field(
        default=0,
        metadata={
            "help": "Number of blocks K to sample from N total blocks for StepMerge (0 = use all blocks). "
            "When K > 0, only K randomly sampled blocks are used for log prob computation. "
            "The same K blocks are used for both old and new log probs to ensure importance sampling consistency."
        },
    )
    use_position_likelihood: bool = field(
        default=False,
        metadata={"help": "Compute a separate position policy loss alongside the StepMerge token loss."},
    )
    position_likelihood_method: str = field(
        default="bradley_terry",
        metadata={"help": "Position scoring method: 'softmax' or 'bradley_terry'"},
    )
    position_likelihood_scope: str = field(
        default="segment",
        metadata={"help": "Competition scope: 'segment' (future segments only) or 'position' (+ within-segment by unmask order)"},
    )
    position_likelihood_confidence: str = field(
        default="max_logit",
        metadata={"help": "Confidence measure: 'max_logit', 'top_prob' (max softmax probability), 'neg_entropy', or 'margin'"},
    )
    position_likelihood_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature tau for position scoring"},
    )
    position_likelihood_lambda: float = field(
        default=1.0,
        metadata={"help": "Weight lambda for position policy loss term (scales position loss relative to token loss)"},
    )
    position_loss_type: str = field(
        default="grpo",
        metadata={"help": "Loss type for position logprobs: 'reinforce', 'grpo', 'dapo', 'gspo', 'gspo_token'"},
    )
    position_clip_eps: float = field(
        default=0.2,
        metadata={"help": "Symmetric clipping epsilon for position loss (for grpo)"},
    )
    position_clip_eps_low: float = field(
        default=0.2,
        metadata={"help": "Asymmetric lower clipping epsilon for position loss (for dapo/gspo)"},
    )
    position_clip_eps_high: float = field(
        default=0.28,
        metadata={"help": "Asymmetric upper clipping epsilon for position loss (for dapo/gspo)"},
    )
    position_averaging_mode: str = field(
        default="token_level",
        metadata={"help": "Averaging mode for position loss: 'token_level' or 'sequence_level'"},
    )
    position_mask_token_clipped: bool = field(
        default=False,
        metadata={
            "help": "When True, tokens whose token importance ratio was clipped are masked out from "
            "position loss, and position loss uses unclipped importance ratios (ratio * advantage, no min). "
            "When False (default), position loss uses its own clipping configured by position_loss_type/position_clip_eps."
        },
    )
    position_confidence_source: str = field(
        default="gt",
        metadata={
            "help": "Confidence source for transitioning tokens in position likelihood: "
            "'gt' (ground truth token logit, default) or 'max' (highest logit, same as competitors). "
            "With 'gt', confidence reflects how sure the model is about the correct token. "
            "With 'max', confidence reflects how sure the model is about its top prediction, regardless of correctness."
        },
    )

    dataset: Optional[str] = field(
        default="gsm8k",
    )

    loss_type: str = field(
        default="grpo",
        metadata={"help": "Loss type: 'reinforce', 'grpo', 'dapo', 'gspo', 'gspo_token'"},
    )

    averaging_mode: str = field(
        default="token_level",
        metadata={"help": "Averaging mode: 'token_level' or 'sequence_level'"},
    )

    clip_eps: float = field(
        default=0.2,
        metadata={"help": "Clipping epsilon for GRPO (symmetric: 1 ± clip_eps)"},
    )

    clip_eps_low: float = field(
        default=0.2,
        metadata={"help": "Lower clipping epsilon for DAPO (1 - clip_eps_low)"},
    )

    clip_eps_high: float = field(
        default=0.28,
        metadata={"help": "Upper clipping epsilon for DAPO (1 + clip_eps_high)"},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.resume_from_checkpoint and self.ignore_data_skip:
            raise ValueError("Resuming rollout reuse requires ignore_data_skip=false")
        if self.dataset not in {"gsm8k", "math"}:
            raise ValueError("dataset must be gsm8k or math")
        if self.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("compute_dtype must be bfloat16, float16 or float32")
        if self.use_vllm:
            raise ValueError("This trainer uses its own diffusion decoder; use_vllm must be false")
        if self.sync_ref_model:
            raise ValueError("Reference synchronization is unsupported for this LoRA trainer; use sync_ref_model=false")
        if self.beta != 0 and self.num_iterations < 2:
            raise ValueError("KL scoring requires num_iterations >= 2 in this trainer")
        if self.num_generations < 2 or self.num_iterations < 1 or self.generation_batch_size < 1:
            raise ValueError("Need num_generations >= 2, num_iterations >= 1 and generation_batch_size >= 1")
        if self.block_length < 1 or self.max_completion_length < 1 or self.max_completion_length % self.block_length:
            raise ValueError("max_completion_length must be a positive multiple of block_length")
        generation_blocks = self.max_completion_length // self.block_length
        if self.diffusion_steps < generation_blocks or self.diffusion_steps % generation_blocks:
            raise ValueError("diffusion_steps must be a positive multiple of the number of generation blocks")
        if self.use_position_likelihood and not self.use_stepmerge:
            raise ValueError("Position likelihood requires use_stepmerge=true")
        if self.use_stepmerge:
            if self.num_stepmerge_blocks < 1 or self.diffusion_steps % self.num_stepmerge_blocks:
                raise ValueError("num_stepmerge_blocks must divide diffusion_steps")
            if not 0 <= self.stepmerge_sample_k_blocks <= self.num_stepmerge_blocks:
                raise ValueError("stepmerge_sample_k_blocks must be between 0 and num_stepmerge_blocks")
            if self.stepmerge_blocks_per_microbatch < 1:
                raise ValueError("stepmerge_blocks_per_microbatch must be positive")
        if self.position_likelihood_temperature <= 0 or self.position_sampling_temperature < 0:
            raise ValueError("Position scoring temperature must be positive; sampling temperature must be nonnegative")
        if self.decouple_position_token_sampling and self.stochastic_position_selection:
            raise ValueError("Choose one position sampling flag; enabling both adds position noise twice")

# Modification notice: Adapted for Mask-Aware Policy Gradients.
