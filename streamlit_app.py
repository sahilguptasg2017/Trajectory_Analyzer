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

# Score extraction patterns
SCORE_PATTERNS = {
    "judge_score": r"JUDGE SCORE:\s*([\d.]+)",
    "rubric_score": r"RUBRIC SCORE:\s*([\d.]+)",
    "verifier_score": r"VERIFIER SCORE:\s*([\d.]+)",
    "multimodal_verifier_score": r"MULTIMODAL VERIFIER SCORE:\s*([\d.]+)",
    "multimodal_step_score": r"MULTIMODAL STEP SCORE:\s*([\d.]+)",
}

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

def extract_detailed_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    """Extract individual score components from the reason field"""
    scores = {}
    reason = rec.get("reason", "") or ""
    
    for score_name, pattern in SCORE_PATTERNS.items():
        match = re.search(pattern, reason, re.IGNORECASE)
        if match:
            try:
                scores[score_name] = float(match.group(1))
            except (ValueError, IndexError):
                scores[score_name] = float("nan")
        else:
            scores[score_name] = float("nan")
    
    return scores

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

        task_id = extract_task_id(rec)
        task_group = extract_task_group(task_id)
        website = extract_website(task_group)

        score = extract_score(rec)
        correct = (score >= threshold) if score == score else False

        # Extract detailed scores
        detailed_scores = extract_detailed_scores(rec)

        tool_calls = extract_tool_calls(rec)
        steps = len(tool_calls)

        counter = Counter()
        for c in tool_calls:
            for k in canonical_tool_keys(c):
                counter[k] += 1

        row_data = {
            "trajectory_index": idx,
            "task_id": task_id,
            "task_group": task_group,
            "website": website,
            "score": score,
            "correct": bool(correct),
            "steps": int(steps),
            "tool_calls_count": dict(counter),
        }
        
        # Add detailed scores
        row_data.update(detailed_scores)
        
        rows.append(row_data)

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
                    "trajectory_index": r["trajectory_index"],
                    "correct": bool(r["correct"]),
                    "tool": tool,
                    "count": int(cnt),
                })
    if not rows:
        return pd.DataFrame(columns=["website", "task_group", "task_id", "trajectory_index", "correct", "tool", "count"])
    out = pd.DataFrame(rows)
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

def create_aligned_histograms(df_correct, df_incorrect, column, title_correct, title_incorrect, nbins=30):
    """Create two histograms with aligned X and Y axes"""
    if df_correct.empty and df_incorrect.empty:
        return None, None
    
    # Combine data to get common range
    all_data = pd.concat([df_correct[column], df_incorrect[column]]).dropna()
    
    if len(all_data) == 0:
        return None, None
    
    # Calculate common x-axis range
    x_min = all_data.min()
    x_max = all_data.max()
    x_range = [x_min - 0.5, x_max + 0.5]
    
    # Create histograms with same bins
    fig_correct = px.histogram(
        df_correct, 
        x=column, 
        nbins=nbins, 
        title=title_correct,
        range_x=x_range
    )
    
    fig_incorrect = px.histogram(
        df_incorrect, 
        x=column, 
        nbins=nbins, 
        title=title_incorrect,
        range_x=x_range
    )
    
    # Calculate common y-axis range
    y_max_correct = 0
    y_max_incorrect = 0
    
    if not df_correct.empty:
        hist_correct, _ = pd.cut(df_correct[column].dropna(), bins=nbins, retbins=True)
        y_max_correct = hist_correct.value_counts().max()
    
    if not df_incorrect.empty:
        hist_incorrect, _ = pd.cut(df_incorrect[column].dropna(), bins=nbins, retbins=True)
        y_max_incorrect = hist_incorrect.value_counts().max()
    
    y_max = max(y_max_correct, y_max_incorrect)
    y_range = [0, y_max * 1.1]  # Add 10% padding
    
    # Update y-axis range for both
    fig_correct.update_yaxes(range=y_range)
    fig_incorrect.update_yaxes(range=y_range)
    
    return fig_correct, fig_incorrect

def display_score_stats(df: pd.DataFrame, score_columns: List[str]):
    """Display statistics for different score types"""
    if not score_columns:
        return
    
    st.subheader("Score Statistics by Type")
    
    score_stats = []
    for col in score_columns:
        if col in df.columns:
            valid_scores = df[col].dropna()
            if len(valid_scores) > 0:
                score_stats.append({
                    "Score Type": col.replace("_", " ").title(),
                    "Mean": f"{valid_scores.mean():.4f}",
                    "Median": f"{valid_scores.median():.4f}",
                    "Std Dev": f"{valid_scores.std():.4f}",
                    "Min": f"{valid_scores.min():.4f}",
                    "Max": f"{valid_scores.max():.4f}",
                    "Count": len(valid_scores)
                })
    
    if score_stats:
        st.dataframe(pd.DataFrame(score_stats), use_container_width=True, hide_index=True)
    
    # Distribution plots for each score type
    if len(score_columns) > 0:
        st.subheader("Score Distributions")
        cols = st.columns(min(3, len(score_columns)))
        for idx, col in enumerate(score_columns):
            if col in df.columns:
                with cols[idx % 3]:
                    fig = px.histogram(
                        df.dropna(subset=[col]), 
                        x=col, 
                        nbins=20,
                        title=col.replace("_", " ").title()
                    )
                    st.plotly_chart(fig, use_container_width=True)

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

    # Identify score columns
    score_columns = [col for col in SCORE_PATTERNS.keys() if col in df.columns]

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

    # Score statistics
    if score_columns:
        display_score_stats(df_f, ["score"] + score_columns)

    # Per-task correctness summary
    st.subheader("Per-task correctness (unique task_id)")
    pt = per_task_summary(df_f)
    st.dataframe(pt, use_container_width=True, height=320)

    # Overall step distributions with aligned axes
    st.subheader("Step-count distributions (overall)")
    left, right = st.columns(2)
    
    df_correct = df_f[df_f["correct"] == True]
    df_incorrect = df_f[df_f["correct"] == False]
    
    fig_correct, fig_incorrect = create_aligned_histograms(
        df_correct, df_incorrect, "steps",
        "Steps in correct trajectories", 
        "Steps in incorrect trajectories"
    )
    
    with left:
        if fig_correct:
            st.plotly_chart(fig_correct, use_container_width=True)
        else:
            st.info("No correct trajectories to display")
    with right:
        if fig_incorrect:
            st.plotly_chart(fig_incorrect, use_container_width=True)
        else:
            st.info("No incorrect trajectories to display")

    # Overall tool usage
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

    # Step distributions for task with aligned axes
    left, right = st.columns(2)
    
    sub_correct = sub[sub["correct"] == True]
    sub_incorrect = sub[sub["correct"] == False]
    
    fig_correct, fig_incorrect = create_aligned_histograms(
        sub_correct, sub_incorrect, "steps",
        f"Steps (correct) — {task}", 
        f"Steps (incorrect) — {task}"
    )
    
    with left:
        if fig_correct:
            st.plotly_chart(fig_correct, use_container_width=True)
        else:
            st.info("No correct trajectories")
    with right:
        if fig_incorrect:
            st.plotly_chart(fig_incorrect, use_container_width=True)
        else:
            st.info("No incorrect trajectories")

    # Tool usage for task
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

    # NEW: Per-trajectory tool usage for selected task
    st.subheader(f"Per-trajectory tool usage — {task}")
    
    # Display trajectories and their tool usage
    for idx, row in sub.iterrows():
        traj_idx = row["trajectory_index"]
        correct_label = "✓ Correct" if row["correct"] else "✗ Incorrect"
        score_label = f"Score: {row['score']:.2f}" if pd.notna(row['score']) else "Score: N/A"
        steps_label = f"Steps: {row['steps']}"
        
        with st.expander(f"Trajectory {traj_idx} — {correct_label} — {score_label} — {steps_label}"):
            tool_counts = row.get("tool_calls_count", {})
            if isinstance(tool_counts, dict) and tool_counts:
                tool_df = pd.DataFrame([
                    {"Tool": tool, "Count": count}
                    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(tool_df, use_container_width=True, hide_index=True)
                
                # Small bar chart
                fig = px.bar(
                    tool_df, 
                    x="Count", 
                    y="Tool", 
                    orientation="h",
                    title=f"Tools used in trajectory {traj_idx}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No tool calls in this trajectory")
            
            # Show detailed scores if available
            if score_columns:
                score_data = []
                for col in score_columns:
                    if col in row and pd.notna(row[col]):
                        score_data.append({
                            "Score Type": col.replace("_", " ").title(),
                            "Value": f"{row[col]:.4f}"
                        })
                if score_data:
                    st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

    # Tool usage table by (task, trajectory, correctness, tool)
    st.subheader("Tool usage table by task and trajectory")
    tb = tool_table(df_f)
    st.dataframe(
        tb.sort_values(["website", "task_group", "task_id", "trajectory_index", "correct", "count"], 
                      ascending=[True, True, True, True, False, False]),
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
        st.download_button("Download tool_usage_by_task_trajectory.csv", tb_csv, file_name="tool_usage_by_task_trajectory.csv", mime="text/csv")

    st.caption("Tool keying: computer_use is counted as computer_use:<action>; web.run is split into web.run:<subcommand> when detectable.")

if __name__ == "__main__":
    main()