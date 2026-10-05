"""Zero-shot GSM8K test prompts used by the experimental evaluator."""
from datasets import load_dataset
from torch.utils.data import Dataset

from parsers import extract_answer_gsm8k

GSM_SYSTEM_PROMPT = "Please reason step by step. Your final answer MUST be in this exact format: <answer> \\boxed{YOUR_ANSWER} </answer>."


class GSM8KDataset(Dataset):
    system_prompt = GSM_SYSTEM_PROMPT
    question_key = "question"

    def __init__(self, tokenizer, add_reasoning=False):
        self.tokenizer = tokenizer
        self.add_reasoning = add_reasoning
        self.dataset = self.load_test_dataset()

    def load_test_dataset(self):
        return load_dataset("gsm8k", "main", split="test")

    def __len__(self):
        return len(self.dataset)

    def create_prompt(self, question):
        messages = [{"role": "user", "content": question + "\n\n" + self.system_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt + "<reasoning>" if self.add_reasoning else prompt

    def reference_answer(self, example):
        return extract_answer_gsm8k(example["answer"])

    def __getitem__(self, index):
        example = self.dataset[index]
        question = example[self.question_key]
        return index, self.create_prompt(question), question, self.reference_answer(example)

    def collate_fn(self, batch):
        indices, prompts, questions, answers = zip(*batch)
        input_ids = self.tokenizer(
            list(prompts), padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "indices": indices, "input_ids": input_ids, "questions": questions,
            "answers": answers, "prompts": prompts,
        }

# Modification notice: Adapted for Mask-Aware Policy Gradients.
