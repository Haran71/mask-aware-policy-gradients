# Mask-Aware Policy Gradients for Diffusion Language Models

Code accompanying **Mask-Aware Policy Gradients for Diffusion Language Models** (COLM 2026).

[Paper](https://arxiv.org/abs/2607.15200)

This repository provides GSM8K and MATH training and evaluation with LLaDA-8B-Instruct. It includes StepMerge likelihood estimation, separate token and position policy losses, NF4/LoRA training, and distributed evaluation.

## Installation

```bash
conda env create -f env.yml
conda activate map
```

Training requires Linux and CUDA GPUs. The environment pins the main dependencies, including the TRL commit used by the implementation.

Model weights and datasets download from Hugging Face on first use. For offline execution, download them beforehand and configure your Hugging Face caches. A local model directory can be supplied through `--model_path`.

LLaDA model loading uses `trust_remote_code=True`.

## Repository structure

```text
map/
├── train.py                 # Training entry point
├── trainer.py               # Diffusion policy optimization
├── config.py                # Training arguments
├── position_likelihood.py   # Position likelihood estimation
├── policy_losses.py         # Policy objectives
├── data_utils.py            # Dataset preparation
├── reward_func.py           # Training rewards
├── math500_utils.py         # Math answer utilities
└── configs/                 # Dataset and distributed configurations

eval/                        # Generation, answer extraction, and scoring
scripts/                     # Training and evaluation launchers
tests/                       # CPU smoke tests
env.yml                      # Environment specification
```

## Training

Run either dataset recipe:

```bash
bash scripts/train_gsm8k.sh
bash scripts/train_math.sh
```

The launchers default to eight GPUs. Adapters and checkpoints are saved under `checkpoints/gsm8k` and `checkpoints/math`.

Override the GPU count and configuration through environment variables and command-line arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  bash scripts/train_gsm8k.sh \
  --output_dir checkpoints/gsm8k_run1
```

The global microbatch size—GPU count multiplied by per-device batch size—must be divisible by `num_generations`, which defaults to 4. Changing GPU count or gradient accumulation changes the effective training batch.

Resume a saved training checkpoint:

```bash
bash scripts/train_math.sh \
  --resume_from_checkpoint checkpoints/math/checkpoint-100
```

Resume with the same dataset, GPU count, batch settings, and `num_iterations`, keeping `ignore_data_skip=false`. Resuming restores the training checkpoint and regenerates rollout buffers, which are not saved. The resumed trajectory can therefore differ from an uninterrupted run.

Inspect available options:

```bash
python -m map.train --help
```

W&B logging is disabled by default. Enable it with `--report_to wandb`. Training launchers target a single machine; `MASTER_PORT` overrides the rendezvous port.


### Datasets

| Task | Training source | Evaluation source |
| --- | --- | --- |
| GSM8K | `openai/gsm8k`, `main`, train split | GSM8K test split |
| MATH | `ankner/math-500`, train split | `HuggingFaceH4/MATH-500`, test split |

The MATH training dataset contains 7,500 examples; the evaluation split contains 500 examples.

## Evaluation

Evaluate the base model:

```bash
bash scripts/eval_gsm8k.sh --load_in_4bit
bash scripts/eval_math.sh --load_in_4bit
```

Evaluate a trained adapter:

```bash
NUM_PROCESSES=4 bash scripts/eval_gsm8k.sh \
  --checkpoint_path checkpoints/gsm8k \
  --load_in_4bit \
  --gen_length 256 \
  --output_dir results/gsm8k_adapter_len256

NUM_PROCESSES=4 bash scripts/eval_math.sh \
  --checkpoint_path checkpoints/math \
  --load_in_4bit \
  --gen_length 256 \
  --output_dir results/math_adapter_len256
```

Evaluation defaults to one GPU, zero-shot prompting, deterministic token decoding, and 128 generated tokens. Diffusion steps default to half the generation length. Omitting `--load_in_4bit` loads the base model in BF16.

Each rank writes a JSON file containing its generations. Compute accuracy after all ranks finish:

```bash
python eval/parse_and_get_acc.py \
  --dataset gsm8k \
  --input_dir results/gsm8k_adapter_len256

python eval/parse_and_get_acc.py \
  --dataset math \
  --input_dir results/math_adapter_len256
```

Use a separate output directory for each checkpoint and generation configuration. The scorer reports correct answers, total examples, and accuracy, and rejects incomplete or inconsistent runs.

Answer extraction and normalization follow the experimental implementation.

## Acknowledgments and license

This implementation builds on [d1](https://github.com/dllm-reasoning/d1), [TRL](https://github.com/huggingface/trl), and [LLaDA](https://github.com/ML-GSAI/LLaDA). Repository organization was informed by [SPG](https://github.com/facebookresearch/SPG).

Code is distributed under the Apache-2.0 license; see `LICENSE`. Model weights and datasets retain their respective licenses.

## Citation

```bibtex
@article{raajesh2026maskaware,
  title={Mask-Aware Policy Gradients for Diffusion Language Models},
  author={Raajesh, Haran and Shah, Kulin and Klivans, Adam and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2607.15200},
  year={2026}
}
```
