import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Optional: native M+ model loader if available
try:
    from modeling_mplus import MPlus  # provided by https://github.com/wangyu-ustc/MemoryLLM
    HAS_MPLUS = True
except Exception:
    MPlus = None
    HAS_MPLUS = False


def normalize_answer(s: str) -> str:
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
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = {}
    for t in prediction_tokens:
        if t in ground_truth_tokens:
            common[t] = min(prediction_tokens.count(t), ground_truth_tokens.count(t))
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


@dataclass
class Example:
    context: str
    question: str
    answers: List[str]


def load_examples(dataset: str, n_eval: int, seed: int) -> List[Example]:
    rng = random.Random(seed)
    if dataset == "squad":
        ds = load_dataset("squad")
        val = ds["validation"]
        # short answers <= 3 tokens
        items = []
        for ex in val:
            ans = ex["answers"]["text"]
            if not ans:
                continue
            if min(len(a.split()) for a in ans) <= 3:
                items.append(
                    Example(context=ex["context"], question=ex["question"], answers=ans)
                )
        rng.shuffle(items)
        return items[:n_eval]
    elif dataset == "nq":
        # natural_questions validation; keep short answers <= 4 tokens
        ds = load_dataset("natural_questions", split="validation")
        items = []
        for ex in ds:
            # Some variants store answers differently; try a few common fields
            answers = []
            if "answers" in ex and isinstance(ex["answers"], dict) and ex["answers"].get("text"):
                answers = ex["answers"]["text"]
            elif "short_answers" in ex and ex["short_answers"]:
                answers = [a.get("text", "") for a in ex["short_answers"] if a.get("text")]
            elif "answer" in ex and ex["answer"]:
                answers = [ex["answer"]]

            if not answers:
                continue
            if min(len(a.split()) for a in answers) <= 4:
                context = ex.get("document_text") or ex.get("context", "")
                question = ex.get("question_text") or ex.get("question", "")
                if context and question:
                    items.append(Example(context=context, question=question, answers=answers))
        rng.shuffle(items)
        return items[:n_eval]
    else:
        raise ValueError("dataset must be 'squad' or 'nq'")


def build_distractor_bank(tokenizer, target_tokens_per_passage=(300, 500), max_bank=5000) -> List[str]:
    ds = load_dataset("squad", split="train")
    passages = []
    low, high = target_tokens_per_passage
    for ex in ds:
        ctx = ex["context"].strip()
        if not ctx:
            continue
        # Split on paragraphs and chunk by tokens to target size
        paras = [p for p in ctx.split("\n") if p.strip()]
        for p in paras:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) == 0:
                continue
            start = 0
            while start < len(ids):
                size = random.randint(low, high)
                chunk_ids = ids[start : start + size]
                start += size
                passages.append(tokenizer.decode(chunk_ids))
                if len(passages) >= max_bank:
                    return passages
    return passages


def build_messages(context: str, distractors: List[str], question: str, system_prompt: str) -> List[Dict[str, str]]:
    user_content = (
        f"Context:\n{context}\n\n"
        f"Additional Passages:\n{('\n\n'.join(distractors)).strip()}\n\n"
        f"Question: {question}\nAnswer:"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_prompt(
    tokenizer,
    context: str,
    distractors: List[str],
    question: str,
    system_prompt: str,
    prefer_chat_template: bool = True,
) -> str:
    # Prefer model-native chat template if available
    if prefer_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = build_messages(context, distractors, question, system_prompt)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback: plain prompt
    prompt = (
        (system_prompt + "\n\n" if system_prompt else "")
        + f"Context:\n{context}\n\n"
        + f"Additional Passages:\n{('\n\n'.join(distractors)).strip()}\n\n"
        + f"Question: {question}\nAnswer:"
    )
    return prompt


def run_generation(model, tokenizer, prompt: str, max_new_tokens=16) -> str:
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
    # Return only the answer portion after last 'Answer:' if present
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip().split("\n")[0].strip()


def eval_budget(
    model,
    tokenizer,
    examples: List[Example],
    distractor_bank: List[str],
    budget_tokens: int,
    window_tokens: int,
    system_prompt: str,
    prefer_chat_template: bool,
) -> Tuple[float, float, int]:
    rng = random.Random(0)
    ems, f1s = [], []
    actual_distance_tokens = []
    for ex in examples:
        # Choose distractors up to budget
        chosen = []
        tok_count = 0
        tries = 0
        while tok_count < budget_tokens and tries < 10000:
            p = rng.choice(distractor_bank)
            tries += 1
            # avoid leaking gold context
            if p in ex.context:
                continue
            ids = tokenizer.encode(p, add_special_tokens=False)
            if tok_count + len(ids) > budget_tokens:
                # take partial to fit budget
                remaining = budget_tokens - tok_count
                if remaining <= 0:
                    break
                ids = ids[:remaining]
                p = tokenizer.decode(ids)
            chosen.append(p)
            tok_count += len(ids)

        # Build prompt and clamp to total window length
        prompt = build_prompt(
            tokenizer,
            ex.context,
            chosen,
            ex.question,
            system_prompt,
            prefer_chat_template=prefer_chat_template,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > window_tokens:
            # Truncate from the front of the prompt but preserve the question tail
            ids = ids[-window_tokens:]
            prompt = tokenizer.decode(ids)
        # Distance is tokens between gold context end and question start approximately = tok_count (after clamp)
        # Recompute actual inserted tokens by tokenizing distractor-only section within the prompt, approximate
        actual = sum(len(tokenizer.encode(d, add_special_tokens=False)) for d in chosen)
        if actual > window_tokens:
            actual = window_tokens
        actual_distance_tokens.append(min(actual, budget_tokens))

        pred = run_generation(model, tokenizer, prompt)
        # Score vs any gold answer
        em = max(exact_match(pred, a) for a in ex.answers)
        f1 = max(f1_score(pred, a) for a in ex.answers)
        ems.append(em)
        f1s.append(f1)

    return float(np.mean(ems)), float(np.mean(f1s)), int(np.median(actual_distance_tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["squad", "nq"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", choices=["none", "4bit"], default="none")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--window", type=int, default=16000)
    parser.add_argument("--budgets", type=str, default="0,2000,8000,16000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a helpful assistant. Answer the question using only the provided context. "
            "Respond with a short answer only."
        ),
        help="System prompt used when building chat prompts",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable using tokenizer.chat_template even if available",
    )
    parser.add_argument("--out", type=str, required=True)
    # M+ specific options
    parser.add_argument("--use_mplus", action="store_true", help="Force loading with modeling_mplus.MPlus if available")
    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "flash_attention_2"], help="Attention impl for MPlus")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype when using MPlus")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Decide loading path
    want_mplus = args.use_mplus or ("mplus" in args.model.lower())
    model = None

    if want_mplus:
        if not HAS_MPLUS:
            raise RuntimeError(
                "Requested MPlus loader but modeling_mplus is not installed. Inside the container, run:\n"
                "  pip install 'git+https://github.com/wangyu-ustc/MemoryLLM.git'\n"
                "Then rerun this script."
            )
        # Map dtype
        if args.dtype == "bfloat16":
            dt = torch.bfloat16
        elif args.dtype == "float16":
            dt = torch.float16
        else:
            dt = torch.float32

        # Load MPlus (quantization via bitsandbytes is not supported here)
        if args.quant == "4bit":
            print("[warn] 4-bit quantization not supported for MPlus loader; using specified dtype instead.")
        model = MPlus.from_pretrained(
            args.model,
            attn_implementation=args.attn_impl,
            torch_dtype=dt,
        )
        # Cast rotary inv_freq properly (per model card) and move to device
        model = model.to(dt)
        # Place long-term memory on CPU and cast to numpy (per model card)
        if hasattr(model, "put_ltm_to_numpy"):
            model.put_ltm_to_numpy()
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        # Standard HF path (supports 4-bit via bitsandbytes)
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

    examples = load_examples(args.dataset, args.n_eval, args.seed)
    distractor_bank = build_distractor_bank(tokenizer)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distance_tokens", "EM", "F1"]) 
        writer.writeheader()
        for b in budgets:
            em, f1, actual = eval_budget(
                model,
                tokenizer,
                examples,
                distractor_bank,
                b,
                args.window,
                args.system_prompt,
                prefer_chat_template=not args.no_chat_template,
            )
            writer.writerow({"distance_tokens": actual, "EM": em, "F1": f1})
            f.flush()
            print(f"Budget {b}: EM={em:.3f} F1={f1:.3f} (actual distance ~{actual})")


if __name__ == "__main__":
    main()
