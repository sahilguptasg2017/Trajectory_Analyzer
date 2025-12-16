import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
WEB_RUN_FIELDS = [
    "search_query", "open", "click", "find", "screenshot", "image_query", "product_query",
    "sports", "finance", "weather", "calculator", "time",
]

def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        s2 = (s.replace("&quot;", '"')
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">"))
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

def extract_score(rec: Dict[str, Any]) -> float:
    for k in ["score", "total_reward", "reward", "episode_reward", "return"]:
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                pass
    return float("nan")

def extract_tool_calls(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    tc = rec.get("tool_calls")
    if isinstance(tc, list) and tc and all(isinstance(x, dict) for x in tc):
        return tc
    seq = rec.get("sequences_str", "") or ""
    return extract_tool_calls_from_sequences(seq)

def parse_jsonl_lines(lines: List[str], threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line = (line or "").strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        task_id = extract_task_id(rec)            # UNIQUE TASK ID (keep as-is)
        task_group = extract_task_group(task_id)  # for website grouping/filtering
        website = extract_website(task_group)

        score = extract_score(rec)
        correct = (score >= threshold) if score == score else False

        tool_calls = extract_tool_calls(rec)
        steps = len(tool_calls)

        counter = Counter()
        for c in tool_calls:
            for k in canonical_tool_keys(c):
                counter[k] += 1

        rows.append({
            "trajectory_index": idx,
            "task_id": task_id,
            "task_group": task_group,
            "website": website,
            "score": score,
            "correct": bool(correct),
            "steps": int(steps),
            "tool_calls_count": dict(counter),
        })

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def load_from_path(path_str: str, threshold: float) -> pd.DataFrame:
    path = Path(path_str).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    return parse_jsonl_lines(lines, threshold)

@st.cache_data(show_spinner=False)
def load_from_upload(upload_bytes: bytes, threshold: float) -> pd.DataFrame:
    text = upload_bytes.decode("utf-8", errors="replace")
    return parse_jsonl_lines(text.splitlines(), threshold)

def tool_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        d = r.get("tool_calls_count") or {}
        if isinstance(d, dict):
            for tool, cnt in d.items():
                rows.append({
                    "website": r["website"],
                    "task_group": r["task_group"],
                    "task_id": r["task_id"],
                    "correct": bool(r["correct"]),
                    "tool": tool,
                    "count": int(cnt),
                })
    if not rows:
        return pd.DataFrame(columns=["website", "task_group", "task_id", "correct", "tool", "count"])
    out = (
        pd.DataFrame(rows)
        .groupby(["website", "task_group", "task_id", "correct", "tool"], as_index=False)["count"]
        .sum()
    )
    return out

def aggregate_tools(sub: pd.DataFrame) -> pd.DataFrame:
    total = Counter()
    for d in sub["tool_calls_count"]:
        if isinstance(d, dict):
            total.update(d)
    return pd.DataFrame(total.most_common(), columns=["tool", "count"])

def per_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["website", "task_group", "task_id"], dropna=False)

    base = g.agg(
        n_trajectories=("trajectory_index", "count"),
        n_correct=("correct", "sum"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
    ).reset_index()

    base["n_incorrect"] = base["n_trajectories"] - base["n_correct"]
    base["frac_correct"] = base["n_correct"] / base["n_trajectories"]
    base["frac_incorrect"] = base["n_incorrect"] / base["n_trajectories"]

    # per-task step stats split by correctness
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

    return out.sort_values(["website", "task_group", "frac_correct", "n_trajectories"], ascending=[True, True, True, False])

def main():
    st.set_page_config(page_title="Trajectory Analyzer", layout="wide")
    st.title("Trajectory Analyzer")

    st.sidebar.header("Input")
    threshold = st.sidebar.number_input("Correctness threshold (score >= threshold)", value=2.0, step=0.1)

    upload = st.sidebar.file_uploader("Upload trajectories.jsonl", type=["jsonl", "txt"])
    path_str = st.sidebar.text_input("...or local path to trajectories.jsonl", value="trajectories.jsonl")

    if upload is not None:
        df = load_from_upload(upload.getvalue(), threshold)
        source_label = f"uploaded file: {upload.name}"
    else:
        try:
            df = load_from_path(path_str, threshold)
            source_label = f"path: {path_str}"
        except Exception as e:
            st.warning(f"Could not read from path. Upload a JSONL file or fix the path.\n\nError: {e}")
            return

    if df.empty:
        st.warning("No trajectories loaded. Check JSONL formatting / tool_call blocks / file content.")
        return

    st.caption(f"Source: {source_label} • Steps = number of <tool_call> blocks • Incorrect if score < {threshold}")

    # Filters
    st.sidebar.header("Filters")
    websites = sorted(df["website"].unique().tolist())
    selected_websites = st.sidebar.multiselect("Website", websites, default=websites)

    df_f = df[df["website"].isin(selected_websites)].copy()

    # Global metrics
    n = len(df_f)
    n_tasks = df_f["task_id"].nunique(dropna=False)
    n_correct = int(df_f["correct"].sum())
    n_incorrect = n - n_correct

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajectories", f"{n}")
    c2.metric("Unique tasks", f"{n_tasks}")
    c3.metric("Correct", f"{n_correct}", f"{(n_correct/n):.1%}" if n else "0%")
    c4.metric("Incorrect", f"{n_incorrect}", f"{(n_incorrect/n):.1%}" if n else "0%")

    # Per-task correctness summary (WHAT YOU ASKED FOR)
    st.subheader("Per-task correctness (unique task_id)")
    pt = per_task_summary(df_f)
    st.dataframe(pt, use_container_width=True, height=320)

    # Overall step distributions (WHAT YOU ASKED FOR)
    st.subheader("Step-count distributions (overall)")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df_f[df_f["correct"] == True], x="steps", nbins=30, title="Steps in correct trajectories")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(df_f[df_f["correct"] == False], x="steps", nbins=30, title="Steps in incorrect trajectories")
        st.plotly_chart(fig, use_container_width=True)

    # Overall tool usage (WHAT YOU ASKED FOR)
    st.subheader("Tool usage (overall)")
    topn = st.slider("Show top N tools/actions", 5, 75, 20)
    left, right = st.columns(2)
    with left:
        tools_c = aggregate_tools(df_f[df_f["correct"] == True]).head(topn)
        fig = px.bar(tools_c, x="count", y="tool", orientation="h", title="Tool/action counts (correct)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tools_c, use_container_width=True, height=260)
    with right:
        tools_i = aggregate_tools(df_f[df_f["correct"] == False]).head(topn)
        fig = px.bar(tools_i, x="count", y="tool", orientation="h", title="Tool/action counts (incorrect)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tools_i, use_container_width=True, height=260)

    # Task drilldown
    st.subheader("Task drilldown (pick a unique task_id)")
    task = st.selectbox("Select task_id", sorted(df_f["task_id"].unique().tolist()))
    sub = df_f[df_f["task_id"] == task].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Trajectories", len(sub))
    c2.metric("Correct", int(sub["correct"].sum()))
    c3.metric("Incorrect", int((~sub["correct"]).sum()))

    left, right = st.columns(2)
    with left:
        fig = px.histogram(sub[sub["correct"] == True], x="steps", nbins=30, title=f"Steps (correct) — {task}")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(sub[sub["correct"] == False], x="steps", nbins=30, title=f"Steps (incorrect) — {task}")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        tc = aggregate_tools(sub[sub["correct"] == True]).head(topn)
        fig = px.bar(tc, x="count", y="tool", orientation="h", title=f"Tools/actions (correct) — {task}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tc, use_container_width=True, height=260)
    with right:
        ti = aggregate_tools(sub[sub["correct"] == False]).head(topn)
        fig = px.bar(ti, x="count", y="tool", orientation="h", title=f"Tools/actions (incorrect) — {task}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ti, use_container_width=True, height=260)

    # Optional: tool usage table by (task, correctness, tool)
    st.subheader("Tool usage table by task and correctness")
    tb = tool_table(df_f)
    st.dataframe(
        tb.sort_values(["website", "task_group", "task_id", "correct", "count"], ascending=[True, True, True, False, False]),
        use_container_width=True,
        height=320
    )

    # Downloads
    st.subheader("Download tables")
    pt_csv = pt.to_csv(index=False).encode("utf-8")
    tb_csv = tb.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download per_task_summary.csv", pt_csv, file_name="per_task_summary.csv", mime="text/csv")
    with d2:
        st.download_button("Download tool_usage_by_task.csv", tb_csv, file_name="tool_usage_by_task.csv", mime="text/csv")

    st.caption("Tool keying: computer_use is counted as computer_use:<action>; web.run is split into web.run:<subcommand> when detectable.")

if __name__ == "__main__":
    main()
