import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("Torch version:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.get_device_name(0))

    # Tiny model smoke test (CPU ok if GPU missing)
    model_name = os.environ.get(
        "SMOKE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Hello! Briefly say hi."
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8)
    print("Generation:", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

