"""Offline CPU smoke checks: python tests/test_trainer.py from the repository root."""
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from trl import ModelConfig, TrlParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map.config import DiffuGRPOConfig
from map.reward_func import correctness_reward_func, correctness_reward_func_math
from map.trainer import DiffuGRPOTrainer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def forward(self, ids):
        hidden = self.embed(ids)
        logits = self.head(hidden + hidden.mean(dim=1, keepdim=True))
        logits[..., 0] = -1e4  # Token 0 is the mask; never generate it.
        return SimpleNamespace(logits=logits)


def check_rollout_resume():
    try:
        DiffuGRPOConfig(use_cpu=True, bf16=False, report_to='none',
                       resume_from_checkpoint='checkpoint-100', ignore_data_skip=True)
    except ValueError as error:
        assert 'ignore_data_skip=false' in str(error)
    else:
        raise AssertionError('Resuming with data skipping disabled must be rejected')
    for start_step in (0, 96, 100):
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.control = SimpleNamespace(should_evaluate=False)
        trainer.state = SimpleNamespace(global_step=start_step)
        trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
        trainer.num_iterations = 12
        trainer._step = 0
        trainer._buffered_inputs = [None] * 4
        calls = []

        def generate(inputs):
            result = {'generation': len(calls), 'prompt': inputs['prompt']}
            calls.append(result)
            return result

        trainer._generate_and_score_completions = generate
        expected = [None] * 4
        # Cover initial filling, reuse and the next scheduled refresh, including
        # checkpoint-100 from the public recipe (100 is not divisible by 12).
        for step in range(start_step, (start_step // 12 + 1) * 12 + 1):
            trainer.state.global_step = step
            for slot in range(4):
                result = trainer._prepare_inputs({'prompt': slot})
                if step == start_step or step % 12 == 0:
                    assert result is not expected[slot]
                    expected[slot] = result
                else:
                    assert result is expected[slot]
                assert result['prompt'] == slot
                assert trainer._step == step * 4 + slot + 1
        assert len(calls) == 8


def main():
    torch.set_num_threads(1)
    check_rollout_resume()
    root = Path(__file__).resolve().parents[1]
    for dataset in ('gsm8k', 'math'):
        args, _ = TrlParser((DiffuGRPOConfig, ModelConfig)).parse_args_and_config(args=[
            '--config', str(root / 'map' / 'configs' / f'{dataset}.yaml'),
            '--bf16', 'false', '--use_cpu', 'true',
        ])
        assert args.use_stepmerge and args.use_position_likelihood
        assert args.loss_type == args.position_loss_type == 'gspo'
        assert args.num_stepmerge_blocks == 32

    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 1, 6, 5, 4, 3]])
    steps = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
    cases = [(True, pos, sampled, iterations, beta)
             for pos in (False, True) for sampled in (False, True)
             for iterations in (1, 2) for beta in (0.0, 0.04)]
    cases += [(False, False, False, 2, 0.0)]
    for stepmerge, position, sampled, iterations, beta in cases:
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.args = SimpleNamespace(
            use_stepmerge=stepmerge, use_position_likelihood=position,
            num_iterations=iterations, num_stepmerge_blocks=2, diffusion_steps=4,
            stepmerge_blocks_per_microbatch=2, stepmerge_sample_k_blocks=int(sampled),
            mask_id=0, cfg_scale=0.0, p_mask_prompt=0.15, fp16=False,
            position_likelihood_method='softmax', position_likelihood_scope='segment',
            position_likelihood_confidence='max_logit', position_likelihood_temperature=1.0,
            position_confidence_source='gt', position_likelihood_lambda=1.0,
            position_mask_token_clipped=False, loss_type='gspo',
            clip_eps_low=0.0003, clip_eps_high=0.0004, averaging_mode='token_level',
            report_to=[],
        )
        trainer.num_iterations = iterations
        trainer.beta = beta
        trainer._step = 0
        trainer._position_loss_args = trainer.args
        trainer.control = SimpleNamespace(should_evaluate=False)
        trainer.accelerator = SimpleNamespace(is_main_process=False, gather_for_metrics=lambda x: x)
        trainer._metrics = {'train': defaultdict(list)}
        torch.manual_seed(7)
        model = ToyModel()
        with torch.no_grad():
            if stepmerge:
                token, pos = trainer._get_per_token_logps_stepmerge(
                    model, ids.unsqueeze(0), 4, [42], steps, 2)
            else:
                token = trainer._get_per_token_logps(model, ids.unsqueeze(0), 4, [42])
                pos = None
        inputs = dict(
            prompt_ids=ids[:, :2], completion_ids=ids[:, 2:],
            completion_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            mask_seeds=[42] * iterations, unmask_steps=steps,
            advantages=torch.tensor([1.0, -0.5]),
            old_per_token_logps=token.repeat(iterations, 1, 1),
            old_pos_logps=pos.repeat(iterations, 1, 1) if pos is not None else None,
            ref_per_token_logps=(token + 0.02).repeat(iterations, 1, 1),
        )
        if sampled:
            inputs['sampled_block_indices'] = [[0]] * iterations
            inputs['sample_masks'] = ((steps > 0) & (steps <= 2)).unsqueeze(0).repeat(iterations, 1, 1)
        loss = trainer.compute_loss(model, inputs)
        loss.backward()
        gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
        assert torch.isfinite(loss) and torch.isfinite(gradients).all()
        assert gradients.abs().sum() > 0
        if stepmerge and position and iterations == 2 and beta == 0:
            # The public recipes use one example per rank. Its logged standard
            # deviations must be zero, rather than undefined sample estimates.
            singleton = {
                key: (value[:, :1] if value.ndim == 3 else value[:1])
                if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            assert torch.isfinite(trainer.compute_loss(model, singleton))
            assert trainer._metrics['train']['advantages/std'][-1] == 0
            prefix = 'gspo_partial' if sampled else 'gspo'
            for component in ('policy', 'position'):
                assert trainer._metrics['train'][f'{component}/{prefix}/importance_ratio_std'][-1] == 0

    with torch.no_grad():
        for decoupled, stochastic in ((False, False), (True, False), (False, True)):
            generated, ordering = trainer.generate(
                model, ids[:, :2], steps=4, gen_length=4, block_length=2,
                temperature=0.9, mask_id=0, decouple_sampling=decoupled,
                stochastic_position_selection=stochastic, position_sampling_temperature=0.5)
            assert torch.equal(generated[:, :2], ids[:, :2])
            assert (generated[:, 2:] != 0).all()
            assert (ordering >= 0).all() and (ordering < 4).all()
    completions = [[{'content': '<answer>42</answer>'}], [{'content': '<answer>7</answer>'}]]
    assert correctness_reward_func([], completions, ['42', '8']) == [2.0, 0.0]
    completions = [[{'content': r'<answer>\boxed{\frac{1}{2}}</answer>'}]]
    assert correctness_reward_func_math([], completions, [r'\boxed{0.5}']) == [2.0]
    print(f'PASS: 2 configs, {len(cases)} loss/gradient cases, 3 generation modes, math rewards, rollout resume')


if __name__ == '__main__':
    main()
