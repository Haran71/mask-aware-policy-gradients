"""Answer extraction and scoring, preserving the experimental evaluator semantics."""
import json
import re

from parser_helper import is_equiv, remove_boxed, last_boxed_only_string


def extract_answer_gsm8k(text):
    """Read a GSM8K reference answer after its #### delimiter."""
    match = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", text)
    return float(match.group(1).replace(",", "")) if match else None


def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
    )


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        parsed_answer = None

        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                parsed_answer = answer_match.group(1).strip()
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
    )

# Modification notice: Adapted for Mask-Aware Policy Gradients.
