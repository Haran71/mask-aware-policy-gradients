"""Score one complete evaluation run; every saved example remains in the denominator."""
import argparse
import json
from pathlib import Path

from parsers import parse_gsm_answers, parse_math_answers


def score_directory(dataset, input_dir):
    files = sorted(Path(input_dir).glob(f"{dataset}_rank*_generations.json"))
    if not files:
        raise ValueError(f"No {dataset} rank generation files in {input_dir}")
    generations = []
    ranks = set()
    expected = None
    config = None
    for path in files:
        payload = json.loads(path.read_text())
        metadata = (payload["dataset"], payload["world_size"], payload["dataset_size"])
        if metadata[0] != dataset:
            raise ValueError(f"Wrong dataset in {path}")
        if expected is None:
            expected, config = metadata, payload["config"]
        if metadata != expected or payload["config"] != config:
            raise ValueError(f"Mixed evaluation configurations in {input_dir}")
        rank = payload["rank"]
        if rank in ranks or not 0 <= rank < metadata[1]:
            raise ValueError(f"Duplicate or invalid rank in {path}")
        ranks.add(rank)
        generations.extend(payload["generations"])
    if ranks != set(range(expected[1])):
        raise ValueError(f"Missing ranks: {sorted(set(range(expected[1])) - ranks)}")
    indices = [item["example_index"] for item in generations]
    if len(indices) != expected[2] or set(indices) != set(range(expected[2])):
        raise ValueError("Incomplete or duplicate examples; refusing to report partial accuracy")
    generations.sort(key=lambda item: item["example_index"])
    parse = parse_gsm_answers if dataset == "gsm8k" else parse_math_answers
    correct, total, _ = parse(json_data={"generations": generations})
    if not total:
        raise ValueError("Cannot score an empty dataset")
    return {"dataset": dataset, "correct": correct, "total": total, "accuracy": correct / total}


def self_test():
    """Offline regression checks for scoring and disjoint evaluation sharding."""
    from eval import shard_indices
    from parsers import extract_answer_gsm8k
    assert extract_answer_gsm8k("Work. #### 1,234") == 1234.0
    cases = [(r"<answer>\boxed{42}</answer>", 42.0), ("<answer>-3</answer>", -3.0)]
    data = {"generations": [{"generations": text, "ground_truth": answer} for text, answer in cases]}
    assert parse_gsm_answers(json_data=data)[:2] == (2, 2)
    data["generations"].append({"generations": "No answer", "ground_truth": 4.0})
    assert parse_gsm_answers(json_data=data)[:2] == (2, 3)
    math_data = {"generations": [{"generations": r"\boxed{\frac{1}{2}}", "ground_truth": "0.5"}]}
    assert parse_math_answers(json_data=math_data)[:2] == (1, 1)
    for size in range(12):
        for workers in range(1, 8):
            indices = [index for rank in range(workers) for index in shard_indices(size, rank, workers)]
            assert sorted(indices) == list(range(size))
    print("Evaluation parser and sharding checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "math"))
    parser.add_argument("--input_dir", help="Directory from one completed eval.py invocation")
    parser.add_argument("--self_test", action="store_true", help="Run small offline parser/sharding checks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or not args.input_dir:
        parser.error("--dataset and --input_dir are required when scoring")
    try:
        result = score_directory(args.dataset, args.input_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# Modification notice: Adapted for Mask-Aware Policy Gradients.
