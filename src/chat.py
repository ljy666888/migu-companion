#!/usr/bin/env python3
"""Interactive tester for the fine-tuned companion model.

The task is "predict the next assistant turn from the conversation so far", so
this keeps a running history and feeds it through the official
``conversation_text`` rendering and ``SYSTEM_PROMPT``. That means what you see
here is exactly what the grader will see, not an approximation.

Commands inside the REPL:
    /reset          clear the conversation
    /history        show the turns sent to the model
    /sample N       load validation sample N (shows the reference answer too)
    /raw            toggle raw model output
    /quit           exit

Usage:
    python src/chat.py                          # fine-tuned adapter
    python src/chat.py --no-adapter             # base model, for comparison
    python src/chat.py --message "最近很累"      # one-shot, no REPL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import build_system_prompt, load_official_runner  # noqa: E402

DEFAULT_BASE = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = ROOT / "outputs" / "qwen25-1.5b-lora"


def load_model(base: str, adapter: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map=device)
    if adapter:
        from peft import PeftModel

        if not (Path(adapter) / "adapter_config.json").is_file():
            raise SystemExit(f"no adapter found at {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # faster inference, same weights
    model.eval()
    return tokenizer, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", "--model", dest="base", default=str(DEFAULT_BASE),
                        help="path to the model directory")
    parser.add_argument("--adapter", default=None,
                        help="optional LoRA adapter directory; omit when the model is already merged")
    parser.add_argument("--no-adapter", action="store_true", help="load the base model only")
    parser.add_argument(
        "--final",
        action="store_true",
        help="test the exact model that ships in the submission image (participant/model)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--message", default=None, help="one-shot message instead of the REPL")
    parser.add_argument("--eval-input", default=str(ROOT / "data" / "eval_input.jsonl"))
    parser.add_argument("--eval-reference", default=str(ROOT / "data" / "eval_reference.jsonl"))
    args = parser.parse_args()

    runner = load_official_runner()
    system_prompt = build_system_prompt(runner)

    if args.final:
        args.base = str(ROOT / "participant" / "model")
        args.adapter = None
        args.no_adapter = True

    adapter = None if (args.no_adapter or not args.adapter) else args.adapter
    print(f"loading {'base model only' if not adapter else 'base + LoRA adapter'} ...", flush=True)
    tokenizer, model = load_model(args.base, adapter, args.device)
    print(f"ready. vram {torch.cuda.memory_allocated()/1024**3:.2f} GB\n", flush=True)

    # Validation samples are loaded lazily so /sample stays instant.
    samples: list[dict] = []
    references: dict[str, dict] = {}
    if Path(args.eval_input).is_file():
        with open(args.eval_input, encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
    if Path(args.eval_reference).is_file():
        with open(args.eval_reference, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    references[row["id"]] = row["target"]

    history: list[dict] = []
    show_raw = False

    def next_turn_id() -> int:
        # Dataset turn ids are 1-based and strictly increasing: 1=user, 2=assistant, ...
        return len(history) + 1

    def respond() -> None:
        if not history:
            print("(empty conversation - type a message first)")
            return
        sample_id = f"test1_000000_t{history[-1]['turn_id']}"
        text = runner.conversation_text({"id": sample_id, "history": history})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)

        if show_raw:
            print("\n--- raw ---")
            print(generated.strip())

        try:
            parsed = runner.extract_object(generated)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[json parse failed: {exc}]")
            print(generated.strip())
            return

        normalized = runner.normalize_prediction("live", parsed)
        print("\n--- reply ---")
        print(normalized["response_text"] or "(empty)")
        print(f"\nemotion : {normalized['emotion_label'] or '(invalid/empty)'}")
        profile = normalized["user_profile"]
        print(f"profile : traits={profile['personality_traits']} interests={profile['interests']} style={profile['style']}")
        print(f"memory  : {normalized['memory_refs']}")

        history.append({"turn_id": next_turn_id(), "role": "assistant",
                        "content": normalized["response_text"]})

    def show_reference(sample: dict) -> None:
        reference = references.get(sample["id"])
        if not reference:
            return
        print("\n--- reference (ground truth) ---")
        print(reference["response_text"])
        print(f"\nemotion : {reference['emotion_label']}")
        profile = reference["user_profile"]
        print(f"profile : traits={profile['personality_traits']} interests={profile['interests']} style={profile['style']}")

    def load_sample(index: int) -> None:
        nonlocal history
        if index < 1 or index > len(samples):
            print(f"sample index must be 1..{len(samples)}")
            return
        sample = samples[index - 1]
        history = [dict(turn) for turn in sample["history"]]
        print(f"\nloaded {sample['id']}  ({len(history)} turns)")
        print("last user message:", history[-1]["content"][:200])
        respond()
        show_reference(sample)

    if args.message:
        history = [{"turn_id": 1, "role": "user", "content": args.message}]
        respond()
        return 0

    print("type a message and press enter. /help for commands.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/reset":
            history, turn_counter = [], 0
            print("(history cleared)")
            continue
        if line == "/history":
            if not history:
                print("(empty)")
            for turn in history:
                print(f"  {turn['turn_id']:>3} {turn['role']:<9} {turn['content'][:110]}")
            continue
        if line == "/raw":
            show_raw = not show_raw
            print(f"(raw output {'on' if show_raw else 'off'})")
            continue
        if line.startswith("/sample"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("usage: /sample <index>")
                continue
            load_sample(int(parts[1]))
            continue
        if line == "/help":
            print("  /reset /history /sample N /raw /quit")
            continue
        if line.startswith("/"):
            print("unknown command; /help")
            continue

        history.append({"turn_id": next_turn_id(), "role": "user", "content": line})
        respond()


if __name__ == "__main__":
    raise SystemExit(main())
