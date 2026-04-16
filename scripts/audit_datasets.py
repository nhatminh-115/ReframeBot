import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def text_stats(values: list[str]) -> dict[str, int]:
    cleaned = [len(v.strip()) for v in values if isinstance(v, str)]
    if not cleaned:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0, "empty_or_ws": 0}
    if len(cleaned) >= 100:
        p95 = int(statistics.quantiles(cleaned, n=100)[94])
    else:
        p95 = max(cleaned)
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "p50": int(statistics.median(cleaned)),
        "p95": p95,
        "max": max(cleaned),
        "empty_or_ws": sum(1 for v in values if not isinstance(v, str) or not v.strip()),
    }


def read_jsonl(path: Path) -> tuple[list[tuple[int, Any]], int, int]:
    rows: list[tuple[int, Any]] = []
    total = 0
    json_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError:
                json_errors += 1
    return rows, total, json_errors


def hash_obj(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def audit_sft(path: Path) -> dict[str, Any]:
    rows, total, json_errors = read_jsonl(path)
    info: dict[str, Any] = {
        "path": str(path),
        "rows": total,
        "json_errors": json_errors,
        "schema_errors": 0,
    }

    top_keys = Counter()
    hashes = Counter()
    message_counts: list[int] = []
    role_dist = Counter()
    missing_user_or_assistant = 0
    empty_contents = 0
    texts: list[str] = []
    malformed_lines: list[int] = []

    for line_no, obj in rows:
        if not isinstance(obj, dict):
            info["schema_errors"] += 1
            malformed_lines.append(line_no)
            continue
        top_keys.update(obj.keys())
        hashes[hash_obj(obj)] += 1

        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            info["schema_errors"] += 1
            malformed_lines.append(line_no)
            continue

        roles = []
        message_counts.append(len(msgs))
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role", "MISSING")
                content = str(m.get("content", ""))
                roles.append(role)
                role_dist[role] += 1
                texts.append(content)
                if not content.strip():
                    empty_contents += 1
            else:
                info["schema_errors"] += 1
        if "user" not in roles or "assistant" not in roles:
            missing_user_or_assistant += 1

    info.update(
        {
            "top_keys": top_keys.most_common(10),
            "duplicate_groups": sum(1 for c in hashes.values() if c > 1),
            "duplicate_rows": sum(c - 1 for c in hashes.values() if c > 1),
            "text_length": text_stats(texts),
            "role_distribution": dict(role_dist),
            "message_count": {
                "min": min(message_counts) if message_counts else 0,
                "p50": int(statistics.median(message_counts)) if message_counts else 0,
                "max": max(message_counts) if message_counts else 0,
            },
            "samples_missing_user_or_assistant": missing_user_or_assistant,
            "empty_message_contents": empty_contents,
            "schema_error_lines_sample": malformed_lines[:20],
        }
    )
    return info


def audit_dpo(path: Path) -> dict[str, Any]:
    rows, total, json_errors = read_jsonl(path)
    info: dict[str, Any] = {
        "path": str(path),
        "rows": total,
        "json_errors": json_errors,
        "schema_errors": 0,
    }
    required = {"prompt", "chosen", "rejected"}
    top_keys = Counter()
    hashes = Counter()
    normalized_group = Counter()
    prompt_to_pairs = defaultdict(set)

    prompt_l: list[str] = []
    chosen_l: list[str] = []
    rejected_l: list[str] = []
    empty_fields = 0
    chosen_eq_rejected = 0
    chosen_shorter = 0
    very_small_gap = 0
    malformed_lines: list[int] = []

    for line_no, obj in rows:
        if not isinstance(obj, dict):
            info["schema_errors"] += 1
            malformed_lines.append(line_no)
            continue
        top_keys.update(obj.keys())
        hashes[hash_obj(obj)] += 1
        if not required.issubset(obj.keys()):
            info["schema_errors"] += 1
            malformed_lines.append(line_no)
            continue

        prompt = str(obj.get("prompt", ""))
        chosen = str(obj.get("chosen", ""))
        rejected = str(obj.get("rejected", ""))
        prompt_l.append(prompt)
        chosen_l.append(chosen)
        rejected_l.append(rejected)

        if not prompt.strip() or not chosen.strip() or not rejected.strip():
            empty_fields += 1
        if chosen.strip() == rejected.strip():
            chosen_eq_rejected += 1
        if len(chosen) < len(rejected):
            chosen_shorter += 1
        if abs(len(chosen) - len(rejected)) <= 5:
            very_small_gap += 1

        nprompt = normalize_text(prompt)
        normalized_group[(nprompt, normalize_text(chosen), normalize_text(rejected))] += 1
        prompt_to_pairs[nprompt].add((normalize_text(chosen), normalize_text(rejected)))

    info.update(
        {
            "top_keys": top_keys.most_common(10),
            "empty_field_rows": empty_fields,
            "chosen_equals_rejected": chosen_eq_rejected,
            "duplicate_groups": sum(1 for c in hashes.values() if c > 1),
            "duplicate_rows": sum(c - 1 for c in hashes.values() if c > 1),
            "normalized_duplicate_groups": sum(1 for c in normalized_group.values() if c > 1),
            "normalized_duplicate_rows": sum(c - 1 for c in normalized_group.values() if c > 1),
            "prompts_with_multiple_preference_pairs": sum(1 for v in prompt_to_pairs.values() if len(v) > 1),
            "chosen_shorter_than_rejected_rows": chosen_shorter,
            "very_small_length_gap_rows": very_small_gap,
            "prompt_length": text_stats(prompt_l),
            "chosen_length": text_stats(chosen_l),
            "rejected_length": text_stats(rejected_l),
            "schema_error_lines_sample": malformed_lines[:20],
        }
    )
    return info


def audit_guardrail(path: Path) -> dict[str, Any]:
    rows, total, json_errors = read_jsonl(path)
    info: dict[str, Any] = {
        "path": str(path),
        "rows": total,
        "json_errors": json_errors,
        "schema_errors": 0,
    }
    required = {"text", "label"}
    top_keys = Counter()
    hashes = Counter()
    by_text_label = Counter()
    text_to_labels = defaultdict(set)
    labels = Counter()
    invalid_labels = 0
    texts: list[str] = []
    malformed_lines: list[int] = []

    for line_no, obj in rows:
        if not isinstance(obj, dict):
            info["schema_errors"] += 1
            malformed_lines.append(line_no)
            continue
        top_keys.update(obj.keys())
        hashes[hash_obj(obj)] += 1
        if set(obj.keys()) != required:
            info["schema_errors"] += 1
            malformed_lines.append(line_no)

        text = str(obj.get("text", ""))
        label = obj.get("label")
        texts.append(text)

        try:
            label_int = int(label)
        except (TypeError, ValueError):
            invalid_labels += 1
            continue

        if label_int not in (0, 1, 2):
            invalid_labels += 1
            continue
        labels[label_int] += 1
        ntext = normalize_text(text)
        text_to_labels[ntext].add(label_int)
        by_text_label[(ntext, label_int)] += 1

    valid_total = sum(labels.values())
    entropy_norm = 0.0
    if valid_total:
        entropy = 0.0
        for k in (0, 1, 2):
            p = labels[k] / valid_total
            if p > 0:
                entropy += -p * math.log2(p)
        entropy_norm = entropy / math.log2(3)

    non_zero = [v for v in (labels[0], labels[1], labels[2]) if v > 0]
    imbalance_ratio = round(max(non_zero) / min(non_zero), 3) if len(non_zero) >= 2 else None

    info.update(
        {
            "top_keys": top_keys.most_common(10),
            "label_counts": dict(labels),
            "invalid_labels": invalid_labels,
            "label_imbalance_ratio_max_over_min": imbalance_ratio,
            "class_balance_entropy_norm_0_to_1": round(entropy_norm, 4),
            "duplicate_groups": sum(1 for c in hashes.values() if c > 1),
            "duplicate_rows": sum(c - 1 for c in hashes.values() if c > 1),
            "normalized_duplicate_rows_same_label": sum(c - 1 for c in by_text_label.values() if c > 1),
            "normalized_text_conflict_across_labels": sum(1 for v in text_to_labels.values() if len(v) > 1),
            "text_length": text_stats(texts),
            "schema_error_lines_sample": malformed_lines[:20],
        }
    )
    return info


def find_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = find_root_from_script()
    parser = argparse.ArgumentParser(description="Audit quality of SFT/DPO/Guardrail JSONL datasets")
    parser.add_argument("--sft", type=Path, default=root / "data" / "dataset.jsonl")
    parser.add_argument("--dpo", type=Path, default=root / "data" / "dataset_dpo.jsonl")
    parser.add_argument("--guardrail", type=Path, default=root / "data" / "guardrail_dataset.jsonl")
    parser.add_argument("--out", type=Path, default=root / "data" / "dataset_audit_report.json")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for name, fn, path in (
        ("sft", audit_sft, args.sft),
        ("dpo", audit_dpo, args.dpo),
        ("guardrail", audit_guardrail, args.guardrail),
    ):
        if path.exists():
            report[name] = fn(path)
        else:
            report[name] = {"path": str(path), "exists": False}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Dataset Audit Complete ===")
    print(f"Report: {args.out}")
    for k in ("sft", "dpo", "guardrail"):
        entry = report.get(k, {})
        if entry.get("exists") is False:
            print(f"- {k}: missing file")
            continue
        print(
            f"- {k}: rows={entry.get('rows', 0)}, json_errors={entry.get('json_errors', 0)}, "
            f"schema_errors={entry.get('schema_errors', 0)}, duplicate_rows={entry.get('duplicate_rows', 0)}"
        )


if __name__ == "__main__":
    main()
