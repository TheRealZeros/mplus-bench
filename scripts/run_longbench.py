import argparse
import json
import os

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--fourbit", action="store_true", help="load in 4-bit")
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = ["hotpotqa", "musique"]

    model_args = {
        "pretrained": args.model,
        "tokenizer": args.model,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "use_fast": True,
        "device_map": "auto",
    }
    if args.fourbit:
        model_args["load_in_4bit"] = True

    hf_model = HFLM(**model_args)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=tasks,
        num_fewshot=0,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Print a short summary
    for t in tasks:
        if t in results.get("results", {}):
            metrics = results["results"][t]
            # Try common keys
            qa_f1 = metrics.get("f1") or metrics.get("qa_f1") or metrics.get("qa_f1_score")
            em = metrics.get("em") or metrics.get("exact_match")
            print(f"{t}: F1={qa_f1} EM={em}")


if __name__ == "__main__":
    main()

