#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

WEB_RUN_FIELDS = [
    "search_query", "open", "click", "find", "screenshot", "image_query", "product_query",
    "sports", "finance", "weather", "calculator", "time",
]

SCORE_FIELDS = ["total_reward", "score", "reward", "episode_reward", "return"]


def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        s2 = (
            s.replace("&quot;", '"')
             .replace("&amp;", "&")
             .replace("&lt;", "<")
             .replace("&gt;", ">")
        )
        try:
            return json.loads(s2)
        except Exception:
            return None


def extract_tool_calls_from_sequences(sequences_str: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in TOOL_CALL_RE.finditer(sequences_str or ""):
        obj = _try_json_loads(m.group(1))
        if obj is not None:
            calls.append(obj)
    return calls


def _extract_name_and_args(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Supports:
      - {"name": "...", "arguments": {...}}
      - {"tool_name": "...", "arguments": {...}}
      - {"type":"function","function":{"name":"...","arguments":"{...}"}} (arguments may be str or dict)
    """
    name = (
        call.get("name")
        or call.get("tool_name")
        or call.get("tool")
        or (call.get("function", {}) or {}).get("name")
        or "UNKNOWN"
    )
    name = str(name)

    args: Any = call.get("arguments", None)
    if args is None and isinstance(call.get("function"), dict):
        args = call["function"].get("arguments", None)

    if isinstance(args, str):
        parsed = _try_json_loads(args)
        args = parsed if isinstance(parsed, dict) else {}
    if not isinstance(args, dict):
        args = {}

    return name, args


def canonical_tool_keys(call: Dict[str, Any]) -> List[str]:
    """
    Returns one or more keys for this tool call.
    - computer_use -> computer_use:<action>
    - web.run/web -> web.run:<subcommand> based on which args are present
    - otherwise -> tool name
    """
    name, args = _extract_name_and_args(call)

    if name == "computer_use":
        action = args.get("action", "UNKNOWN")
        return [f"{name}:{action}"]

    if name in ("web.run", "web"):
        keys = []
        for f in WEB_RUN_FIELDS:
            if args.get(f) not in (None, [], {}, "", False):
                keys.append(f"{name}:{f}")
        return keys if keys else [name]

    return [name]


def extract_task_id(rec: Dict[str, Any]) -> str:
    initial = rec.get("initial_config", {}) or {}
    task_id = initial.get("id") or initial.get("task_id") or rec.get("task_id") or "UNKNOWN_TASK"
    return str(task_id)


def extract_task_group(task_id: str) -> str:
    parts = task_id.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1]) or task_id
    return task_id


def extract_website(task_group: str) -> str:
    return (task_group.split("_", 1)[0] or "UNKNOWN_WEBSITE").strip()


def _to_float_or_nan(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def extract_all_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in SCORE_FIELDS:
        out[k] = _to_float_or_nan(rec.get(k))
    return out


def choose_score_for_correctness(scores: Dict[str, float], preferred: str) -> Tuple[float, str]:
    """
    Pick preferred score for correctness; if missing/NaN, fallback to first available among SCORE_FIELDS.
    """
    if preferred in scores and scores[preferred] == scores[preferred]:
        return scores[preferred], preferred

    for k in SCORE_FIELDS:
        v = scores.get(k, float("nan"))
        if v == v:
            return v, k

    return float("nan"), "NONE"


def extract_tool_calls(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    tc = rec.get("tool_calls")
    if isinstance(tc, list) and tc and all(isinstance(x, dict) for x in tc):
        return tc
    seq = rec.get("sequences_str", "") or ""
    return extract_tool_calls_from_sequences(seq)


def parse_jsonl(path: Path, threshold: float, correctness_score_field: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            task_id = extract_task_id(rec)
            task_group = extract_task_group(task_id)
            website = extract_website(task_group)

            scores = extract_all_scores(rec)
            score_value, score_key_used = choose_score_for_correctness(scores, correctness_score_field)
            correct = (score_value >= threshold) if score_value == score_value else False  # NaN -> incorrect

            tool_calls = extract_tool_calls(rec)
            steps = len(tool_calls)

            counter = Counter()
            for c in tool_calls:
                for k in canonical_tool_keys(c):
                    counter[k] += 1

            row = {
                "trajectory_index": idx,
                "task_id": task_id,
                "task_group": task_group,
                "website": website,

                # correctness
                "correctness_score_field": correctness_score_field,
                "score_key_used": score_key_used,
                "score_value": score_value,
                "correct": bool(correct),

                # trajectory info
                "steps": int(steps),
                "tool_calls_count": dict(counter),
            }

            # include all score types as separate columns
            for k, v in scores.items():
                row[k] = v

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_per_task(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["website", "task_group", "task_id"], dropna=False)

    base = g.agg(
        n_trajectories=("trajectory_index", "count"),
        n_correct=("correct", "sum"),
    ).reset_index()

    base["n_incorrect"] = base["n_trajectories"] - base["n_correct"]
    base["frac_correct"] = base["n_correct"] / base["n_trajectories"]
    base["frac_incorrect"] = base["n_incorrect"] / base["n_trajectories"]

    # score stats per score field
    for k in SCORE_FIELDS:
        base[f"avg_{k}"] = g[k].mean().values
        base[f"median_{k}"] = g[k].median().values

    # stats for score_value (the one used for correctness)
    base["avg_score_value_used"] = g["score_value"].mean().values
    base["median_score_value_used"] = g["score_value"].median().values

    # step stats split by correctness within each task
    def _steps_stats(sub: pd.DataFrame) -> Dict[str, Any]:
        c = sub[sub["correct"] == True]["steps"]
        i = sub[sub["correct"] == False]["steps"]
        return {
            "avg_steps_correct": float(c.mean()) if len(c) else float("nan"),
            "median_steps_correct": float(c.median()) if len(c) else float("nan"),
            "avg_steps_incorrect": float(i.mean()) if len(i) else float("nan"),
            "median_steps_incorrect": float(i.median()) if len(i) else float("nan"),
        }

    stats = g.apply(_steps_stats).apply(pd.Series).reset_index()
    out = base.merge(stats, on=["website", "task_group", "task_id"], how="left")

    return out.sort_values(
        ["website", "task_group", "frac_correct", "n_trajectories"],
        ascending=[True, True, True, False],
    )


def tool_usage_overall(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def agg(sub: pd.DataFrame) -> pd.DataFrame:
        total = Counter()
        for d in sub["tool_calls_count"]:
            if isinstance(d, dict):
                total.update(d)
        return pd.DataFrame(total.most_common(), columns=["tool", "count"])

    return agg(df[df["correct"] == True]), agg(df[df["correct"] == False])


def tool_usage_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format table with one row per (trajectory, tool).
    This is the requested "per trajectory per task" view.
    """
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d = r.get("tool_calls_count") or {}
        if not isinstance(d, dict):
            continue
        for tool, cnt in d.items():
            rows.append({
                "website": r["website"],
                "task_group": r["task_group"],
                "task_id": r["task_id"],
                "trajectory_index": int(r["trajectory_index"]),
                "correct": bool(r["correct"]),
                "steps": int(r["steps"]),
                "score_value": float(r["score_value"]) if r["score_value"] == r["score_value"] else float("nan"),
                "score_key_used": r.get("score_key_used", ""),
                "tool": tool,
                "count": int(cnt),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "website", "task_group", "task_id", "trajectory_index", "correct", "steps",
            "score_value", "score_key_used", "tool", "count"
        ])

    return pd.DataFrame(rows).sort_values(
        ["website", "task_group", "task_id", "trajectory_index", "correct", "count"],
        ascending=[True, True, True, True, False, False],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to trajectories.jsonl")
    ap.add_argument("--out", default="analysis_out", type=str, help="Output directory")
    ap.add_argument("--threshold", default=2.0, type=float, help="Correct if score_value >= threshold")
    ap.add_argument(
        "--correctness_score_field",
        default="total_reward",
        choices=SCORE_FIELDS,
        help="Which score field to use for correctness (fallback to others if missing).",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_jsonl(in_path, threshold=args.threshold, correctness_score_field=args.correctness_score_field)
    if df.empty:
        raise SystemExit("No trajectories parsed. Check input file path / JSONL formatting.")

    df.to_csv(out_dir / "trajectories_parsed.csv", index=False)

    per_task = summarize_per_task(df)
    per_task.to_csv(out_dir / "per_task_summary.csv", index=False)

    tool_c, tool_i = tool_usage_overall(df)
    tool_c.to_csv(out_dir / "tool_usage_overall_correct.csv", index=False)
    tool_i.to_csv(out_dir / "tool_usage_overall_incorrect.csv", index=False)

    # NEW: per-trajectory tool usage (requested)
    tpt = tool_usage_per_trajectory(df)
    tpt.to_csv(out_dir / "tool_usage_per_trajectory.csv", index=False)

    n = len(df)
    n_tasks = df["task_id"].nunique(dropna=False)
    n_correct = int(df["correct"].sum())

    print(f"Loaded trajectories: {n}")
    print(f"Unique tasks (task_id): {n_tasks}")
    print(f"Correctness field: {args.correctness_score_field} (fallback if missing)")
    print(f"Correct (>= {args.threshold}): {n_correct} ({(n_correct/n if n else 0):.1%})")
    print(f"Outputs written to: {out_dir.resolve()}")
    print("Files:")
    print(" - trajectories_parsed.csv")
    print(" - per_task_summary.csv")
    print(" - tool_usage_overall_correct.csv")
    print(" - tool_usage_overall_incorrect.csv")
    print(" - tool_usage_per_trajectory.csv")


if __name__ == "__main__":
    main()
