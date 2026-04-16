import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_sft(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        normalized_messages: list[dict[str, str]] = []
        has_user = False
        has_assistant = False
        valid = True
        for msg in messages:
            if not isinstance(msg, dict):
                valid = False
                break
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if not role or not content:
                valid = False
                break
            if role == "user":
                has_user = True
            if role == "assistant":
                has_assistant = True
            normalized_messages.append({"role": role, "content": content})
        if not valid or not has_user or not has_assistant:
            continue

        key = json.dumps(normalized_messages, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"messages": normalized_messages})
    return out


def dedupe_dpo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not all(k in row for k in ("prompt", "chosen", "rejected")):
            continue
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected:
            continue
        if chosen == rejected:
            continue

        key = (normalize_text(prompt), normalize_text(chosen), normalize_text(rejected))
        if key in seen:
            continue
        seen.add(key)

        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


def dedupe_guardrail(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Keep only texts that map to exactly one label. Conflict texts are separated for manual review.
    text_to_labels: dict[str, set[int]] = {}
    text_to_raw: dict[str, str] = {}

    for row in rows:
        text = str(row.get("text", "")).strip()
        label = row.get("label")
        if not text:
            continue
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        if label_int not in (0, 1, 2):
            continue

        ntext = normalize_text(text)
        text_to_labels.setdefault(ntext, set()).add(label_int)
        if ntext not in text_to_raw:
            text_to_raw[ntext] = text

    clean: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for ntext, labels in text_to_labels.items():
        if len(labels) > 1:
            conflicts.append({"text": text_to_raw[ntext], "labels": sorted(labels)})
            continue
        clean.append({"text": text_to_raw[ntext], "label": list(labels)[0]})

    clean.sort(key=lambda x: (int(x["label"]), normalize_text(str(x["text"]))))
    conflicts.sort(key=lambda x: normalize_text(str(x["text"])))
    return clean, conflicts


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Create deduplicated copies of SFT/DPO/Guardrail datasets")
    parser.add_argument("--sft", type=Path, default=root / "data" / "dataset.jsonl")
    parser.add_argument("--dpo", type=Path, default=root / "data" / "dataset_dpo.jsonl")
    parser.add_argument("--guardrail", type=Path, default=root / "data" / "guardrail_dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "cleaned")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = read_jsonl(args.sft)
    dpo_rows = read_jsonl(args.dpo)
    guard_rows = read_jsonl(args.guardrail)

    sft_clean = dedupe_sft(sft_rows)
    dpo_clean = dedupe_dpo(dpo_rows)
    guard_clean, guard_conflicts = dedupe_guardrail(guard_rows)

    sft_out = args.out_dir / "dataset.sft.cleaned.jsonl"
    dpo_out = args.out_dir / "dataset.dpo.cleaned.jsonl"
    guard_out = args.out_dir / "dataset.guardrail.cleaned.jsonl"
    conflicts_out = args.out_dir / "dataset.guardrail.conflicts.jsonl"

    write_jsonl(sft_out, sft_clean)
    write_jsonl(dpo_out, dpo_clean)
    write_jsonl(guard_out, guard_clean)
    write_jsonl(conflicts_out, guard_conflicts)

    print("=== Dataset Dedupe Complete ===")
    print(f"SFT: {args.sft} -> {sft_out} ({len(sft_rows)} -> {len(sft_clean)})")
    print(f"DPO: {args.dpo} -> {dpo_out} ({len(dpo_rows)} -> {len(dpo_clean)})")
    print(f"Guardrail: {args.guardrail} -> {guard_out} ({len(guard_rows)} -> {len(guard_clean)})")
    print(f"Guardrail conflicts: {conflicts_out} ({len(guard_conflicts)})")


if __name__ == "__main__":
    main()
