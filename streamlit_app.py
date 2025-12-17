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


def _to_float_or_nan(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def extract_all_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    return {k: _to_float_or_nan(rec.get(k)) for k in SCORE_FIELDS}


def choose_score_for_correctness(scores: Dict[str, float], preferred: str) -> Tuple[float, str]:
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


def parse_jsonl_lines(lines: List[str]) -> pd.DataFrame:
    """
    Parse only; correctness is computed later in the app so the user can change threshold / score field.
    """
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line = (line or "").strip()
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
            "steps": int(steps),
            "tool_calls_count": dict(counter),
        }
        row.update(scores)
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    return parse_jsonl_lines(lines)


@st.cache_data(show_spinner=False)
def load_from_upload(upload_bytes: bytes) -> pd.DataFrame:
    text = upload_bytes.decode("utf-8", errors="replace")
    return parse_jsonl_lines(text.splitlines())


def apply_correctness(df: pd.DataFrame, threshold: float, correctness_score_field: str) -> pd.DataFrame:
    df = df.copy()

    # pick score_value + score_key_used per row (with fallback)
    score_values = []
    score_keys = []
    for _, r in df.iterrows():
        scores = {k: r.get(k, float("nan")) for k in SCORE_FIELDS}
        v, key_used = choose_score_for_correctness(scores, correctness_score_field)
        score_values.append(v)
        score_keys.append(key_used)

    df["correctness_score_field"] = correctness_score_field
    df["score_key_used"] = score_keys
    df["score_value"] = score_values

    df["correct"] = df["score_value"].apply(lambda x: (x >= threshold) if x == x else False)
    return df


def per_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["website", "task_group", "task_id"], dropna=False)

    base = g.agg(
        n_trajectories=("trajectory_index", "count"),
        n_correct=("correct", "sum"),
    ).reset_index()

    base["n_incorrect"] = base["n_trajectories"] - base["n_correct"]
    base["frac_correct"] = base["n_correct"] / base["n_trajectories"]
    base["frac_incorrect"] = base["n_incorrect"] / base["n_trajectories"]

    # score stats per field
    for k in SCORE_FIELDS:
        base[f"avg_{k}"] = g[k].mean().values
        base[f"median_{k}"] = g[k].median().values

    # score used for correctness
    base["avg_score_value_used"] = g["score_value"].mean().values
    base["median_score_value_used"] = g["score_value"].median().values

    # step stats split by correctness per task
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


def aggregate_tools(sub: pd.DataFrame) -> pd.DataFrame:
    total = Counter()
    for d in sub["tool_calls_count"]:
        if isinstance(d, dict):
            total.update(d)
    return pd.DataFrame(total.most_common(), columns=["tool", "count"])


def tool_usage_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requested: per-trajectory tool usage (long format).
    """
    rows = []
    for _, r in df.iterrows():
        d = r.get("tool_calls_count") or {}
        if isinstance(d, dict):
            for tool, cnt in d.items():
                rows.append({
                    "website": r["website"],
                    "task_group": r["task_group"],
                    "task_id": r["task_id"],
                    "trajectory_index": int(r["trajectory_index"]),
                    "correct": bool(r["correct"]),
                    "steps": int(r["steps"]),
                    "score_value": float(r["score_value"]) if r.get("score_value") == r.get("score_value") else float("nan"),
                    "score_key_used": r.get("score_key_used", ""),
                    "tool": tool,
                    "count": int(cnt),
                })
    if not rows:
        return pd.DataFrame(columns=[
            "website", "task_group", "task_id", "trajectory_index", "correct", "steps",
            "score_value", "score_key_used", "tool", "count"
        ])
    return pd.DataFrame(rows)


def _aligned_step_histograms(df: pd.DataFrame, title_left: str, title_right: str):
    """
    Two histograms (correct vs incorrect) with aligned X and Y axis ranges.
    Bin size is 1 step for comparability.
    """
    c = df[df["correct"] == True]["steps"]
    i = df[df["correct"] == False]["steps"]

    if len(df) == 0:
        x_min, x_max = 0, 1
    else:
        x_min = int(df["steps"].min())
        x_max = int(df["steps"].max())
        if x_min == x_max:
            x_max = x_min + 1

    # y max based on exact integer step counts (bin size 1)
    c_max = int(c.value_counts().max()) if len(c) else 0
    i_max = int(i.value_counts().max()) if len(i) else 0
    y_max = max(c_max, i_max, 1)
    y_range = [0, int(y_max * 1.1) + 1]

    # histogram binning: center bins on integers
    xbins = dict(start=x_min - 0.5, end=x_max + 0.5, size=1)

    fig_c = px.histogram(df[df["correct"] == True], x="steps", title=title_left, nbins=(x_max - x_min + 1))
    fig_c.update_traces(xbins=xbins)
    fig_c.update_xaxes(range=[x_min - 0.5, x_max + 0.5])
    fig_c.update_yaxes(range=y_range)

    fig_i = px.histogram(df[df["correct"] == False], x="steps", title=title_right, nbins=(x_max - x_min + 1))
    fig_i.update_traces(xbins=xbins)
    fig_i.update_xaxes(range=[x_min - 0.5, x_max + 0.5])
    fig_i.update_yaxes(range=y_range)

    return fig_c, fig_i


def score_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall score stats split by correctness for each score field.
    """
    rows = []
    for k in SCORE_FIELDS + ["score_value"]:
        for label, sub in [("correct", df[df["correct"] == True]), ("incorrect", df[df["correct"] == False])]:
            s = sub[k] if k in sub.columns else pd.Series(dtype=float)
            rows.append({
                "score_field": k,
                "subset": label,
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "count_non_nan": int(s.notna().sum()) if len(s) else 0,
            })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Trajectory Analyzer", layout="wide")
    st.title("Trajectory Analyzer")

    st.sidebar.header("Input")
    threshold = st.sidebar.number_input("Correctness threshold (>=)", value=2.0, step=0.1)
    correctness_score_field = st.sidebar.selectbox(
        "Correctness score field",
        options=SCORE_FIELDS,
        index=SCORE_FIELDS.index("total_reward") if "total_reward" in SCORE_FIELDS else 0,
    )

    upload = st.sidebar.file_uploader("Upload trajectories.jsonl", type=["jsonl", "txt"])
    path_str = st.sidebar.text_input("...or local path to trajectories.jsonl", value="trajectories.jsonl")

    if upload is not None:
        df_raw = load_from_upload(upload.getvalue())
        source_label = f"uploaded file: {upload.name}"
    else:
        try:
            df_raw = load_from_path(path_str)
            source_label = f"path: {path_str}"
        except Exception as e:
            st.warning(f"Could not read from path. Upload a JSONL file or fix the path.\n\nError: {e}")
            return

    if df_raw.empty:
        st.warning("No trajectories loaded. Check JSONL formatting / tool_call blocks / file content.")
        return

    df = apply_correctness(df_raw, threshold=threshold, correctness_score_field=correctness_score_field)

    st.caption(
        f"Source: {source_label} • Steps = number of <tool_call> blocks • "
        f"Incorrect if {correctness_score_field} < {threshold} (fallback if missing)"
    )

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

    # Per-task correctness summary (now with multiple score stats)
    st.subheader("Per-task correctness (unique task_id) + score stats")
    pt = per_task_summary(df_f)
    st.dataframe(pt, use_container_width=True, height=320)

    # Score stats (overall)
    st.subheader("Score statistics (overall, split by correctness)")
    st.dataframe(score_stats_table(df_f), use_container_width=True, height=260)

    # Step distributions (aligned axes)
    st.subheader("Step-count distributions (overall, aligned axes)")
    left, right = st.columns(2)
    fig_c, fig_i = _aligned_step_histograms(
        df_f,
        title_left="Steps in correct trajectories",
        title_right="Steps in incorrect trajectories",
    )
    with left:
        st.plotly_chart(fig_c, use_container_width=True)
    with right:
        st.plotly_chart(fig_i, use_container_width=True)

    # Overall tool usage (totals still useful)
    st.subheader("Tool usage (overall totals)")
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

    # Task step distributions (aligned axes)
    st.markdown("#### Step distributions for selected task (aligned axes)")
    left, right = st.columns(2)
    fig_c, fig_i = _aligned_step_histograms(
        sub,
        title_left=f"Steps (correct) — {task}",
        title_right=f"Steps (incorrect) — {task}",
    )
    with left:
        st.plotly_chart(fig_c, use_container_width=True)
    with right:
        st.plotly_chart(fig_i, use_container_width=True)

    # Task tool usage totals (for the selected task)
    st.markdown("#### Tool usage totals for selected task")
    left, right = st.columns(2)
    with left:
        tc = aggregate_tools(sub[sub["correct"] == True]).head(topn)
        fig = px.bar(tc, x="count", y="tool", orientation="h", title=f"Tools/actions totals (correct) — {task}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tc, use_container_width=True, height=260)
    with right:
        ti = aggregate_tools(sub[sub["correct"] == False]).head(topn)
        fig = px.bar(ti, x="count", y="tool", orientation="h", title=f"Tools/actions totals (incorrect) — {task}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ti, use_container_width=True, height=260)

    # NEW: per-trajectory tool usage table (requested)
    st.subheader("Tool usage per trajectory (selected task)")
    tpt = tool_usage_per_trajectory(sub)
    st.dataframe(
        tpt.sort_values(["trajectory_index", "correct", "count"], ascending=[True, False, False]),
        use_container_width=True,
        height=320,
    )

    # Optional: pick a trajectory and show its tool breakdown
    st.markdown("#### Trajectory drilldown (single trajectory tool breakdown)")
    traj = st.selectbox("Select trajectory_index", sorted(sub["trajectory_index"].unique().tolist()))
    one = tpt[tpt["trajectory_index"] == traj].sort_values("count", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    row = sub[sub["trajectory_index"] == traj].iloc[0]
    c1.metric("Trajectory index", int(traj))
    c2.metric("Correct", str(bool(row["correct"])))
    c3.metric("Steps", int(row["steps"]))
    c4.metric("Score used", f"{row['score_value']:.3f}" if row["score_value"] == row["score_value"] else "NaN")

    fig = px.bar(one.head(topn), x="count", y="tool", orientation="h", title=f"Tool usage — trajectory {traj}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(one, use_container_width=True, height=260)

    # Downloads
    st.subheader("Download tables")
    pt_csv = pt.to_csv(index=False).encode("utf-8")
    tpt_csv = tool_usage_per_trajectory(df_f).to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download per_task_summary.csv", pt_csv, file_name="per_task_summary.csv", mime="text/csv")
    with d2:
        st.download_button("Download tool_usage_per_trajectory.csv", tpt_csv, file_name="tool_usage_per_trajectory.csv", mime="text/csv")

    st.caption(
        "Tool keying: computer_use is counted as computer_use:<action>; "
        "web.run is split into web.run:<subcommand> when detectable. "
        "Histograms use bin size 1 and aligned axes for comparison."
    )


if __name__ == "__main__":
    main()
