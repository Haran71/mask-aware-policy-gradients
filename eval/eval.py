"""Save zero-shot GSM8K/MATH generations on one GPU or with torchrun."""
import argparse
import json
import os
from pathlib import Path
import random
import time


def shard_indices(size, rank, world_size):
    """Disjoint shards, including empty shards when workers outnumber examples."""
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset size, rank, or world size")
    return range(rank, size, world_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint_path", default="", help="Optional local or Hugging Face LoRA adapter")
    parser.add_argument("--output_dir", required=True, help="A fresh directory for this evaluation run")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=None, help="Defaults to gen_length // 2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--decouple_sampling", action="store_true")
    parser.add_argument("--position_sampling_temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.diffusion_steps is None:
        args.diffusion_steps = args.gen_length // 2
    if min(args.batch_size, args.gen_length, args.block_length, args.diffusion_steps) <= 0:
        parser.error("batch_size, gen_length, block_length, and diffusion_steps must be positive")
    if args.gen_length % args.block_length or args.diffusion_steps % (args.gen_length // args.block_length):
        parser.error("gen_length must divide into blocks and diffusion_steps must divide by the block count")
    if min(args.temperature, args.position_sampling_temperature) < 0:
        parser.error("Sampling temperatures must be nonnegative")
    if args.position_sampling_temperature and not args.decouple_sampling:
        parser.error("position_sampling_temperature requires --decouple_sampling")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from generate import generate
    from gsm8k import GSM8KDataset
    from math500 import MATH500Dataset

    if not torch.cuda.is_available():
        parser.error("Generation requires CUDA; --help and result scoring work without a GPU")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{args.dataset}_rank{rank}_generations.json"
    if filename.exists():
        raise FileExistsError(f"Refusing to overwrite {filename}; use a fresh --output_dir")
    dtype = torch.bfloat16
    if args.checkpoint_path:
        config_path = Path(args.checkpoint_path) / "adapter_config.json"
        if config_path.is_file():
            adapter_config = json.loads(config_path.read_text())
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                adapter_config.get("torch_dtype"), torch.bfloat16
            )
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.load_in_4bit:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            ),
            device_map={"": local_rank},
        )
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=dtype)
    model.eval()
    dataset_class = GSM8KDataset if args.dataset == "gsm8k" else MATH500Dataset
    dataset = dataset_class(tokenizer, add_reasoning=args.add_reasoning)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=shard_indices(len(dataset), rank, world_size), collate_fn=dataset.collate_fn,
    )
    generations = []
    start = time.perf_counter()
    for batch in tqdm(dataloader, disable=rank != 0):
        result = generate(
            model, batch["input_ids"].to(device), tokenizer,
            gen_length=args.gen_length, steps=args.diffusion_steps, block_length=args.block_length,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
            decouple_sampling=args.decouple_sampling,
            position_sampling_temperature=args.position_sampling_temperature,
        )
        texts = tokenizer.batch_decode(result[:, -args.gen_length:], skip_special_tokens=False)
        for index, prompt, question, answer, text in zip(
            batch["indices"], batch["prompts"], batch["questions"], batch["answers"], texts
        ):
            generations.append({
                "example_index": index, "question": question, "prompt_input": prompt,
                "generations": text, "ground_truth": answer,
            })
    torch.cuda.synchronize()
    payload = {
        "generations": generations, "config": vars(args), "dataset": args.dataset,
        "rank": rank, "world_size": world_size, "dataset_size": len(dataset),
        "metrics": {"total_processed": len(generations), "wall_time_seconds": time.perf_counter() - start},
    }
    with filename.open("x") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Rank {rank}: saved {len(generations)} / {len(dataset)} examples to {filename}")


if __name__ == "__main__":
    main()

# Modification notice: Adapted for Mask-Aware Policy Gradients.
