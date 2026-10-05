"""Train the GSM8K or MATH adapter: python -m map.train --config map/configs/gsm8k.yaml."""

import torch
from accelerate import PartialState
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from trl import ModelConfig, TrlParser

from .config import DiffuGRPOConfig
from .data_utils import get_gsm8k_questions, get_math_questions, set_random_seed
from .reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from .trainer import DiffuGRPOTrainer


def main(grpo_config, model_config):
    if not torch.cuda.is_available():
        raise RuntimeError("Training the NF4 LLaDA model requires a CUDA GPU.")
    if not model_config.load_in_4bit or not model_config.use_peft:
        raise ValueError("This entry point requires load_in_4bit=true and use_peft=true.")
    set_random_seed(grpo_config.seed)
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [xmlcount_reward_func, soft_format_reward_func,
                            strict_format_reward_func, int_reward_func, correctness_reward_func]
    else:
        dataset = get_math_questions("train")
        reward_functions = [correctness_reward_func_math, boxed_and_answer_tags_format_reward]
    dataset = dataset.shuffle(seed=grpo_config.seed)

    compute_dtype = getattr(torch, grpo_config.compute_dtype)
    grpo_config.fp16 = compute_dtype == torch.float16
    grpo_config.bf16 = compute_dtype == torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs = {}
    if model_config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_config.attn_implementation
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=quantization,
        device_map={"": PartialState().local_process_index},
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
        use_dora=grpo_config.dora,
    )
    trainer = DiffuGRPOTrainer(
        model=model,
        args=grpo_config,
        peft_config=peft_config,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint or None)
    trainer.save_model(grpo_config.output_dir)


if __name__ == "__main__":
    grpo_config, model_config = TrlParser((DiffuGRPOConfig, ModelConfig)).parse_args_and_config()
    main(grpo_config, model_config)

# Modification notice: Adapted for Mask-Aware Policy Gradients.
