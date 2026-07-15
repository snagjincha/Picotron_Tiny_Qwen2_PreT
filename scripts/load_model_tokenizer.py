import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a tokenizer and causal language model.")
    parser.add_argument(
        "--model-name",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, pretraining practice!",
        help="Prompt used for a quick generation smoke test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Number of tokens to generate for the smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"model_name: {args.model_name}")
    print(f"tokenizer_vocab_size: {len(tokenizer)}")
    print(f"model_parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("generated_text:")
    print(decoded)


if __name__ == "__main__":
    main()

