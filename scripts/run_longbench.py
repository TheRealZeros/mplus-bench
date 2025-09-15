import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Optional: native M+ model loader
try:
    from modeling_mplus import MPlus  # provided by https://github.com/wangyu-ustc/MemoryLLM
    HAS_MPLUS = True
except Exception:
    MPlus = None
    HAS_MPLUS = False


def normalize_answer(s: str) -> str:
    import re

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    common = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


@dataclass
class LBExample:
    input_text: str
    answers: List[str]


def load_longbench(task: str, source: str, path: Optional[str], n_eval: int, seed: int, shuffle: bool = True) -> List[LBExample]:
    rng = random.Random(seed)
    items: List[LBExample] = []
    if source == "hf":
        # Try common variants for THUDM/LongBench subsets
        candidates = []
        if task == "hotpotqa":
            candidates = [
                ("THUDM/LongBench", "hotpotqa"),
                ("THUDM/LongBench", "hotpot_qa"),
            ]
        elif task == "musique":
            candidates = [
                ("THUDM/LongBench", "musique"),
                ("THUDM/LongBench", "MuSiQue"),
            ]
        else:
            raise ValueError("task must be 'hotpotqa' or 'musique'")

        ds = None
        err = None
        for name, subset in candidates:
            try:
                ds = load_dataset(name, subset, split="validation")
                break
            except Exception as e:
                err = e
                continue
        if ds is None:
            raise RuntimeError(f"Could not load LongBench subset {task} from HF: {err}")

        for ex in ds:
            # Common LongBench fields
            input_text = ex.get("input") or ex.get("context") or ex.get("document") or ""
            if not input_text:
                # As a fallback, combine available text fields
                parts = []
                for k in ["context", "question", "passage", "document_text"]:
                    if ex.get(k):
                        parts.append(str(ex[k]))
                input_text = "\n\n".join(parts)
            answers = []
            if isinstance(ex.get("answers"), list) and ex.get("answers"):
                answers = [a if isinstance(a, str) else str(a) for a in ex["answers"]]
            elif ex.get("answer"):
                if isinstance(ex["answer"], list):
                    answers = [a if isinstance(a, str) else str(a) for a in ex["answer"]]
                else:
                    answers = [str(ex["answer"])]
            if input_text and answers:
                items.append(LBExample(input_text=input_text, answers=answers))
    else:
        # source == file: read JSONL with fields {input, answers|answer}
        if not path:
            raise ValueError("--data_path is required when --source file")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                inp = obj.get("input") or obj.get("context") or ""
                answers = obj.get("answers") or obj.get("answer") or []
                if isinstance(answers, str):
                    answers = [answers]
                if inp and answers:
                    items.append(LBExample(input_text=inp, answers=[str(a) for a in answers]))

    if shuffle:
        rng.shuffle(items)
    return items[:n_eval]


def clamp_prompt_ids(tokenizer, user_content: str, system_prompt: str, window_tokens: int, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > window_tokens:
            ids = ids[-window_tokens:]
        return tokenizer.decode(ids)
    else:
        prefix = (system_prompt + "\n\n") if system_prompt else ""
        text = prefix + user_content
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > window_tokens:
            ids = ids[-window_tokens:]
        return tokenizer.decode(ids)


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip().split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["hotpotqa", "musique"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", choices=["none", "4bit"], default="none")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--window", type=int, choices=[8000, 16000], default=16000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--source", choices=["hf", "file"], default="hf")
    parser.add_argument("--data_path", type=str, default=None, help="Path to JSONL when --source file")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a helpful assistant. Answer the question using only the provided context. "
            "Respond with a short answer only."
        ),
    )
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    # M+ loader options
    parser.add_argument("--use_mplus", action="store_true")
    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "flash_attention_2"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load model
    want_mplus = args.use_mplus or ("mplus" in args.model.lower())
    if want_mplus:
        if not HAS_MPLUS:
            raise RuntimeError(
                "Requested MPlus loader but modeling_mplus is not installed. Inside the container, run:\n"
                "  pip install 'git+https://github.com/wangyu-ustc/MemoryLLM.git'\n"
                "Then rerun this script."
            )
        if args.dtype == "bfloat16":
            dt = torch.bfloat16
        elif args.dtype == "float16":
            dt = torch.float16
        else:
            dt = torch.float32
        model = MPlus.from_pretrained(args.model, attn_implementation=args.attn_impl, torch_dtype=dt)
        model = model.to(dt)
        if hasattr(model, "put_ltm_to_numpy"):
            model.put_ltm_to_numpy()
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        model_kwargs = {}
        if args.quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if torch.cuda.is_available() and args.quant == "none":
            model = model.cuda()

    # Load data
    examples = load_longbench(args.task, args.source, args.data_path, args.n_eval, args.seed, shuffle=not args.no_shuffle)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    use_chat = not args.no_chat_template
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "EM", "F1"]) 
        writer.writeheader()

        ems, f1s = [] , []
        for i, ex in enumerate(examples):
            # The LongBench record usually contains the full context + question; append an explicit answer cue
            user_content = ex.input_text.strip()
            if not user_content.endswith("Answer:"):
                user_content = user_content.rstrip() + "\nAnswer:"

            prompt = clamp_prompt_ids(tokenizer, user_content, args.system_prompt, args.window, use_chat)
            pred = run_generation(model, tokenizer, prompt, args.max_new_tokens)
            em = max(exact_match(pred, a) for a in ex.answers)
            f1 = max(f1_score(pred, a) for a in ex.answers)
            ems.append(em)
            f1s.append(f1)
            writer.writerow({"index": i, "EM": em, "F1": f1})
            f.flush()

    mean_em = float(np.mean(ems)) if ems else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"LongBench {args.task} (n={len(examples)} @ {args.window}): EM={mean_em:.3f} F1={mean_f1:.3f}")


if __name__ == "__main__":
    main()
