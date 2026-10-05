"""Zero-shot MATH-500 test prompts used by the experimental evaluator."""
from datasets import load_dataset
from gsm8k import GSM8KDataset

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class MATH500Dataset(GSM8KDataset):
    system_prompt = MATH500_SYSTEM_PROMPT
    question_key = "problem"

    def load_test_dataset(self):
        return load_dataset("HuggingFaceH4/MATH-500", split="test")

    def reference_answer(self, example):
        return example["answer"]

# Modification notice: Adapted for Mask-Aware Policy Gradients.
