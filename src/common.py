"""Shared helpers for the Migu digital-human companion task.

Prompt construction and sample iteration are delegated to the official
``official/participant/run_inference.py`` module so that the SFT training input
is byte-identical to what the grader feeds the model at inference time.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = PROJECT_ROOT / "official" / "participant" / "run_inference.py"

TARGET_FIELDS = ("response_text", "emotion_label", "user_profile", "memory_refs")

_runner_cache: dict[str, object] = {}


def load_official_runner(path=None):
    """Import the official runner so its constants are reused, never copied."""
    resolved = Path(path or DEFAULT_RUNNER).resolve()
    key = str(resolved)
    if key not in _runner_cache:
        spec = importlib.util.spec_from_file_location("official_run_inference", resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official runner: {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _runner_cache[key] = module
    return _runner_cache[key]


def build_system_prompt(runner) -> str:
    """Mirror the ``SYSTEM_PROMPT.format(...)`` call inside run_inference.py main().

    One deliberate deviation: ``run_inference.py`` interpolates ``EMOTIONS`` with
    ``", ".join(runner.EMOTIONS)`` and ``EMOTIONS`` is a ``set``, so the official
    prompt lists the emotion labels in a different order on every process launch
    (PYTHONHASHSEED randomisation). The label *set* is what matters, so we sort it
    to keep the training data reproducible and verifiable. The three profile
    groups are already ``sorted()`` by the official runner.
    """
    return runner.SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(runner.EMOTIONS)),
        personality=", ".join(sorted(runner.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(runner.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(runner.PROFILE_LABELS["style"])),
    )


def synth_sample_id(prefix: str, conversation_index: int, target_user_turn_id: int) -> str:
    """Mimic the id shape seen in the official sample file (``test1_000001_t5``)."""
    return f"{prefix}{conversation_index:06d}_t{target_user_turn_id}"


def build_history(turns, target_user_turn_id):
    """Turns up to and including the target user turn, ordered by turn_id.

    Truncation here is what prevents the answer from leaking into the input:
    the reference assistant turn always sits at target_user_turn_id + 1.
    """
    history = [turn for turn in turns if turn.get("turn_id", 0) <= target_user_turn_id]
    history.sort(key=lambda turn: turn["turn_id"])
    return history


def validate_point(turns, target_user_turn_id, assistant_turn_id, target):
    """Return a skip reason string, or None when the prediction point is usable."""
    if not isinstance(target_user_turn_id, int) or not isinstance(assistant_turn_id, int):
        return "missing turn ids"
    if assistant_turn_id != target_user_turn_id + 1:
        return "assistant_response_turn_id is not target_user_turn_id + 1"
    if not isinstance(target, dict):
        return "missing target object"
    missing = [field for field in TARGET_FIELDS if field not in target]
    if missing:
        return f"target missing fields: {missing}"

    by_id = {turn.get("turn_id"): turn for turn in turns}
    user_turn = by_id.get(target_user_turn_id)
    response_turn = by_id.get(assistant_turn_id)
    if user_turn is None or user_turn.get("role") != "user":
        return "target turn is not present as a user turn"
    if response_turn is None or response_turn.get("role") != "assistant":
        return "response turn is not present as an assistant turn"

    history = build_history(turns, target_user_turn_id)
    if not history or history[-1].get("role") != "user":
        return "history does not end with a user turn"
    if response_turn["content"] != target["response_text"]:
        return "response_text does not match the reference assistant turn"
    return None


def iter_prediction_points(row):
    """Yield (target_user_turn_id, assistant_turn_id, target) for one dataset row."""
    for point in row.get("prediction_points") or []:
        yield (
            point.get("target_user_turn_id"),
            point.get("assistant_response_turn_id"),
            point.get("target"),
        )


def render_input(runner, sample_id, history) -> str:
    """Call the official ``conversation_text`` so the rendering cannot drift."""
    return runner.conversation_text({"id": sample_id, "history": history})


def serialize_target(target) -> str:
    """Match how run_inference.py writes each submission line."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
