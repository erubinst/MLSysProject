"""
KV Cache Compression Results — Figure Generator
Produces:
    - Figure 1: Static vs. Dynamic grouped bar chart (accuracy)
    - Figure 2: Static average accuracy line plot by granularity and scoring backend
    - Figure 3: TFLOPs vs. Accuracy scatter (static vs. dynamic)
    - Figure 4: Throughput vs. Accuracy scatter (static vs. dynamic)
    - Figure 5: Full results table (LaTeX + CSV)
    - Figure 6: Latency breakdown — stacked bar (prefill vs. decode, dynamic only)
    - Figure 7: Per-task heatmap (granularity × scoring backend, one panel per task)
    - Figure 8: Scoring backend comparison — grouped bars
    - Figure 9: Static-only latency comparison — horizontal bar chart
    - Figure 10: Static-only throughput comparison — horizontal bar chart
    - Figure 11: Static vs. Dynamic average score comparison (Quest/H2O only)
    - Figure 12: Static vs. Dynamic latency comparison (Quest/H2O only)

Usage:
  python make_figures.py \
      [--static path/to/static_results.csv] \
      --dynamic path/to/summary_dynamic.csv \
      --outdir  figures/
"""

import argparse, csv, os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── display labels ────────────────────────────────────────────────────────────
GRANULARITY_LABELS = {
    "clusterattn": "Block-density\n(clusterattn)",
    "pagekv":      "Page-level\n(pagekv)",
    "tokenkv":     "Token-level\n(tokenkv)",
}
SCORING_LABELS = {
    "snapkv":       "SnapKV",
    "quest_bounds": "Quest bounds",
    "h2o":          "H2O",
    "recon":        "Reconstruction",
    "expected_attention": "Exp. Attention",
    "random":       "Random",
}

STATIC_COLOR  = "#333333"
DYNAMIC_COLOR = "#990000"
TIMEOUT_COLOR = "#bbbbbb"

GRANULARITY_COLORS = {
    "clusterattn": "#7a2e2e",
    "pagekv": "#1f6f78",
    "tokenkv": "#b87900",
}

GRANULARITY_LINE_LABELS = {
    "clusterattn": "Cluster / block-density",
    "pagekv": "Page",
    "tokenkv": "Token",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def parse_static_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = r["Method"].strip()
            if not method:
                continue
            rows.append({
                "method":      method,
                "gov_report":  parse_float(r.get("gov_report")),
                "hotpotqa":    parse_float(r.get("hotpotqa")),
                "lcc":         parse_float(r.get("lcc")),
                "qasper":      parse_float(r.get("qasper")),
                "avg":         parse_float(r.get("Average")),
                "peak_gpu":    parse_float(r.get("Peak GPU (GB)")),
                "kv_mb":       parse_float(r.get("KV Cache (MB)")),
                "latency":     parse_float(r.get("Avg Latency (s)")),
                "prefill_lat": parse_float(r.get("Avg Prefill Latency (s)")),
                "decode_lat":  parse_float(r.get("Avg Decode Latency (s)")),
                "throughput":  parse_float(r.get("Throughput (tok/s)")),
                "tflops":      parse_float(r.get("Profiled TFLOPs")),
                "tflops_s":    parse_float(r.get("Profiled TFLOPs/s")),
                "mode":        "static" if "_static" in method else "baseline",
                "timed_out":   False,
            })
    return rows

def parse_static_readme_table(path="README.md"):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    markers = [
        "### Merged Static Result: `v1_20260427_030527` + `v9_20260503_152041`",
        "### Current benchmark snapshot",
    ]
    marker = next((m for m in markers if m in text), None)
    if marker is None:
        raise ValueError(f"Could not find a static benchmark table in {path}")

    sub = text[text.index(marker):]
    lines = [ln for ln in sub.splitlines() if ln.startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"Could not find a markdown table under '{marker}' in {path}")

    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        parts = [c.strip() for c in ln.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        r = dict(zip(header, parts))
        method = r["Method"].strip()
        if not method:
            continue
        rows.append({
            "method":      method,
            "gov_report":  parse_float(r.get("gov_report")),
            "hotpotqa":    parse_float(r.get("hotpotqa")),
            "lcc":         parse_float(r.get("lcc")),
            "qasper":      parse_float(r.get("qasper")),
            "avg":         parse_float(r.get("Average")),
            "peak_gpu":    parse_float(r.get("Peak GPU (GB)")),
            "kv_mb":       parse_float(r.get("KV Cache (MB)")),
            "latency":     parse_float(r.get("Avg Latency (s)")),
            "prefill_lat": parse_float(r.get("Avg Prefill Latency (s)")),
            "decode_lat":  parse_float(r.get("Avg Decode Latency (s)")),
            "throughput":  parse_float(r.get("Throughput (tok/s)")),
            "tflops":      parse_float(r.get("Profiled TFLOPs")),
            "tflops_s":    parse_float(r.get("Profiled TFLOPs/s")),
            "mode":        "static" if "_static" in method else "baseline",
            "timed_out":   False,
        })
    return rows

def parse_dynamic_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = r["Method"].strip()
            if not method:
                continue
            avg = parse_float(r.get("Average"))
            rows.append({
                "method":      method,
                "gov_report":  parse_float(r.get("gov_report")),
                "hotpotqa":    parse_float(r.get("hotpotqa")),
                "lcc":         parse_float(r.get("lcc")),
                "qasper":      parse_float(r.get("qasper")),
                "avg":         avg,
                "peak_gpu":    parse_float(r.get("Peak GPU (GB)")),
                "kv_mb":       parse_float(r.get("KV Cache (MB)")),
                "latency":     parse_float(r.get("Avg Latency (s)")),
                "prefill_lat": parse_float(r.get("Avg Prefill Latency (s)")),
                "decode_lat":  parse_float(r.get("Avg Decode Latency (s)")),
                "throughput":  parse_float(r.get("Throughput (tok/s)")),
                "tflops":      parse_float(r.get("Profiled TFLOPs")),
                "tflops_s":    parse_float(r.get("Profiled TFLOPs/s")),
                "mode":        "dynamic",
                "timed_out":   avg is None,
            })
    return rows

def extract_parts(method):
    for gran in ["clusterattn", "pagekv", "tokenkv"]:
        if method.startswith(gran):
            rest = method[len(gran)+1:]
            for sfx in ["_static", "_dynamic"]:
                rest = rest.replace(sfx, "")
            return gran, rest
    return None, None

def fmt(val, decimals=2):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

def _save(fig, outdir, name):
    html = os.path.join(outdir, f"{name}.html")
    fig.write_html(html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved {html}")

def _comparison_rows(static_rows, dynamic_rows, scorings=("quest_bounds", "h2o")):
    rows = []
    for gran in ["clusterattn", "pagekv", "tokenkv"]:
        for scoring in scorings:
            s = next((r for r in static_rows if extract_parts(r["method"]) == (gran, scoring)), None)
            d = next((r for r in dynamic_rows if extract_parts(r["method"]) == (gran, scoring)), None)
            if s is None:
                continue
            rows.append({
                "gran": gran,
                "scoring": scoring,
                "label": f"{gran}<br>{SCORING_LABELS[scoring]}",
                "static_avg": s["avg"],
                "dynamic_avg": d["avg"] if d else None,
                "static_lat": s["latency"],
                "dynamic_lat": d["latency"] if d else None,
                "timed_out": d["timed_out"] if d else True,
            })
    return rows

def add_reference_lines(fig, static_rows):
    baseline = next((r for r in static_rows if r["method"] == "baseline"), None)
    snapkv   = next((r for r in static_rows if r["method"] == "snapkv_static"), None)
    if baseline and baseline["avg"] is not None:
        fig.add_hline(y=baseline["avg"], line=dict(color="#888888", width=1, dash="dot"))
        fig.add_annotation(x=1.0, xref="paper", y=baseline["avg"] + 0.15, yref="y",
            text=f"Full precision ({baseline['avg']:.1f})", showarrow=False,
            xanchor="right", font=dict(size=11, color="#888888"))
    if snapkv and snapkv["avg"] is not None:
        fig.add_hline(y=snapkv["avg"], line=dict(color="#555555", width=1, dash="dash"))
        fig.add_annotation(x=1.0, xref="paper", y=snapkv["avg"] - 0.5, yref="y",
            text=f"SnapKV ({snapkv['avg']:.1f})", showarrow=False,
            xanchor="right", font=dict(size=11, color="#555555"))

def base_layout(fig, title, width=980, height=460):
    fig.update_layout(
        template="plotly_white", width=width, height=height,
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=60, r=25, t=60, b=75),
        font=dict(family="Arial, sans-serif", size=12),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Static vs. Dynamic grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig1_static_vs_dynamic(static_rows, dynamic_rows, outdir):
    dyn_scorings  = ["quest_bounds", "h2o", "random"]
    granularities = ["clusterattn", "pagekv", "tokenkv"]

    pairs = {}
    for gran in granularities:
        for scoring in dyn_scorings:
            s = next((r for r in static_rows  if extract_parts(r["method"]) == (gran, scoring)), None)
            d = next((r for r in dynamic_rows if extract_parts(r["method"]) == (gran, scoring)), None)
            if s:
                pairs[(gran, scoring)] = {
                    "static":    s["avg"],
                    "dynamic":   d["avg"] if d else None,
                    "timed_out": d["timed_out"] if d else False,
                }

    y_min, y_max = 25, 46
    n_gran = len(granularities)
    n_scoring = len(dyn_scorings)
    group_width   = 0.7
    bar_w         = group_width / (n_scoring * 2 + n_scoring - 1) * 0.9
    pair_gap      = bar_w * 0.25
    group_spacing = 1.0
    xtick_positions, xtick_labels = [], []
    shapes, annotations = [], []

    for gi, gran in enumerate(granularities):
        group_center = gi * group_spacing
        pair_centers = np.linspace(
            group_center - group_width/2 + bar_w,
            group_center + group_width/2 - bar_w,
            n_scoring,
        )
        for pi, scoring in enumerate(dyn_scorings):
            data  = pairs.get((gran, scoring), {})
            s_val = data.get("static")
            d_val = data.get("dynamic")
            timed = data.get("timed_out", False)
            cx    = pair_centers[pi]
            xs    = cx - bar_w/2 - pair_gap/2
            xd    = cx + bar_w/2 + pair_gap/2

            if s_val is not None:
                shapes.append(dict(type="rect", x0=xs-bar_w/2, x1=xs+bar_w/2,
                    y0=y_min, y1=s_val, line=dict(width=0),
                    fillcolor=STATIC_COLOR, opacity=0.85))
                annotations.append(dict(x=xs, y=s_val+0.25, text=f"{s_val:.1f}",
                    showarrow=False, xanchor="center", yanchor="bottom",
                    font=dict(size=10, color=STATIC_COLOR)))

            if timed or d_val is None:
                shapes.append(dict(type="rect", x0=xd-bar_w/2, x1=xd+bar_w/2,
                    y0=y_min, y1=y_min+0.7, line=dict(width=0),
                    fillcolor=TIMEOUT_COLOR, opacity=0.6))
                annotations.append(dict(x=xd, y=y_min+0.9, text="T/O",
                    showarrow=False, xanchor="center", yanchor="bottom",
                    font=dict(size=10, color=TIMEOUT_COLOR)))
            else:
                shapes.append(dict(type="rect", x0=xd-bar_w/2, x1=xd+bar_w/2,
                    y0=y_min, y1=d_val, line=dict(width=0),
                    fillcolor=DYNAMIC_COLOR, opacity=0.85))
                annotations.append(dict(x=xd, y=d_val+0.25, text=f"{d_val:.1f}",
                    showarrow=False, xanchor="center", yanchor="bottom",
                    font=dict(size=10, color=DYNAMIC_COLOR)))
                if s_val is not None:
                    delta = d_val - s_val
                    sign  = "+" if delta >= 0 else ""
                    annotations.append(dict(x=cx, y=max(s_val, d_val)+1.2,
                        text=f"{sign}{delta:.1f}", showarrow=False, xanchor="center",
                        yanchor="bottom", font=dict(size=10, color="#555555")))

            xtick_positions.append(cx)
            xtick_labels.append(SCORING_LABELS[scoring])

        annotations.append(dict(x=group_center, y=-0.18, xref="x", yref="paper",
            text=GRANULARITY_LABELS[gran].replace("\n", " "),
            showarrow=False, xanchor="center", yanchor="top",
            font=dict(size=13, color="#222222")))

        if gi < n_gran - 1:
            shapes.append(dict(type="line",
                x0=group_center+group_spacing/2, x1=group_center+group_spacing/2,
                y0=y_min, y1=y_max,
                line=dict(color="#cccccc", width=1, dash="dash")))

    fig = go.Figure()
    add_reference_lines(fig, static_rows)
    fig.update_layout(shapes=shapes, annotations=annotations)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color=STATIC_COLOR,  size=12), name="Static",   hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color=DYNAMIC_COLOR, size=12), name="Dynamic",  hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color=TIMEOUT_COLOR, size=12, symbol="square"), name="Timed out", hoverinfo="skip"))
    fig.update_xaxes(tickmode="array", tickvals=xtick_positions, ticktext=xtick_labels,
        range=[-0.6, (n_gran-1)*group_spacing+0.6], showgrid=False, zeroline=False)
    fig.update_yaxes(title="Average accuracy", range=[y_min, y_max],
        showgrid=True, gridcolor="#eeeeee", zeroline=False)
    base_layout(fig, "Figure 1: Static vs. Dynamic Execution — Average LongBench Accuracy")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5))
    _save(fig, outdir, "fig1_static_vs_dynamic")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Static average accuracy line plot
# ─────────────────────────────────────────────────────────────────────────────

def fig2_static_accuracy_line(static_rows, outdir):
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    scorings = ["snapkv", "quest_bounds", "h2o", "recon", "expected_attention", "random"]
    x = list(range(len(scorings)))
    xlabels = [SCORING_LABELS[s] for s in scorings]

    row_by_parts = {}
    for row in static_rows:
        gran, scoring = extract_parts(row["method"])
        if gran in granularities and scoring in scorings and row.get("avg") is not None:
            row_by_parts[(gran, scoring)] = row

    baseline = next((r for r in static_rows if r["method"] == "baseline"), None)
    baseline_y = baseline["avg"] if baseline else None

    fig = go.Figure()

    if baseline_y is not None:
        fig.add_hline(
            y=baseline_y,
            line=dict(color="#2b2b2b", width=2, dash="dash"),
            annotation_text=f"Baseline {baseline_y:.2f}",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#2b2b2b"),
        )

    best_global = None
    for gran in granularities:
        ys = []
        hover = []
        methods = []
        for scoring in scorings:
            row = row_by_parts.get((gran, scoring))
            ys.append(row["avg"] if row else None)
            methods.append(row["method"] if row else "")
            hover.append(
                f"{row['method']}<br>{GRANULARITY_LINE_LABELS[gran]} + {SCORING_LABELS[scoring]}<br>"
                f"Average accuracy: {row['avg']:.2f}<extra></extra>"
                if row else ""
            )

        color = GRANULARITY_COLORS[gran]
        fig.add_trace(go.Scatter(
            x=x,
            y=ys,
            mode="lines+markers",
            connectgaps=False,
            name=GRANULARITY_LINE_LABELS[gran],
            line=dict(color=color, width=3, dash="dot"),
            marker=dict(size=10, color=color, line=dict(color="white", width=1.2)),
            customdata=methods,
            hovertemplate=hover,
        ))

        label_shift = {
            "clusterattn": 24,
            "pagekv": -24,
            "tokenkv": -42,
        }[gran]
        for xi, yi in zip(x, ys):
            if yi is None:
                continue
            xshift = 0
            if xi in (4, 5):
                xshift = {
                    "clusterattn": -18,
                    "pagekv": 18,
                    "tokenkv": 0,
                }[gran]
            fig.add_annotation(
                x=xi,
                y=yi,
                text=f"{yi:.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom" if label_shift > 0 else "top",
                xshift=xshift,
                yshift=label_shift,
                font=dict(size=10, color=color),
                bgcolor="rgba(255,255,255,0.65)",
                borderpad=1,
            )

        valid_points = [(idx, val) for idx, val in enumerate(ys) if val is not None]
        if not valid_points:
            continue
        best_idx, best_val = max(valid_points, key=lambda item: item[1])
        best_row = row_by_parts[(gran, scorings[best_idx])]
        if best_global is None or best_val > best_global["avg"]:
            best_global = {"gran": gran, "idx": best_idx, "avg": best_val, "row": best_row}

        fig.add_trace(go.Scatter(
            x=[best_idx],
            y=[best_val],
            mode="markers",
            marker=dict(
                symbol="star",
                size=17,
                color=color,
                line=dict(color="#111111", width=1.8),
            ),
            showlegend=False,
            hovertemplate=(
                f"Best {GRANULARITY_LINE_LABELS[gran]}<br>{best_row['method']}<br>"
                f"Average accuracy: {best_val:.2f}<extra></extra>"
            ),
        ))

        offset = {
            "clusterattn": (-95, -85),
            "pagekv": (-85, 82),
            "tokenkv": (112, -72),
        }[gran]
        fig.add_annotation(
            x=best_idx,
            y=best_val,
            text=(
                f"<span style='color:{color}'><b>Best {GRANULARITY_LINE_LABELS[gran]}</b><br>"
                f"{METHOD_DISPLAY.get(best_row['method'], best_row['method'])} ({best_val:.2f})</span>"
            ),
            showarrow=True,
            arrowhead=2,
            ax=offset[0],
            ay=offset[1],
            font=dict(size=11, color=color),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=color,
            borderwidth=1,
        )

    if best_global is not None:
        fig.add_annotation(
            x=0.98,
            xref="paper",
            y=1,
            yref="paper",
            text=f"Overall best: {best_global['row']['method']} ({best_global['avg']:.2f})",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=12, color="#111111"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#111111",
            borderwidth=1,
        )

    y_values = [
        row["avg"]
        for row in row_by_parts.values()
        if row.get("avg") is not None
    ]
    if baseline_y is not None:
        y_values.append(baseline_y)
    y_min = max(20, min(y_values) - 3.0) if y_values else 30
    y_max = max(y_values) + 3.0 if y_values else 45

    fig.update_xaxes(
        title="Eviction / scoring backend",
        tickmode="array",
        tickvals=x,
        ticktext=xlabels,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title="Average LongBench accuracy",
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
    )
    base_layout(
        fig,
        "Figure 2: Static Average Accuracy by Granularity and Eviction Backend",
        width=960,
        height=660,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=70, r=170, t=140, b=105),
    )
    _save(fig, outdir, "fig2_static_accuracy_line")


# ─────────────────────────────────────────────────────────────────────────────
# Figures 3 & 4: Scatter plots (shared helper)
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_plotly(static_rows, dynamic_rows, xkey, xlabel):
    gran_markers = {"clusterattn": "circle", "pagekv": "square", "tokenkv": "triangle-up"}
    gran_display = {"clusterattn": "Block-density", "pagekv": "Page-level", "tokenkv": "Token-level"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color=STATIC_COLOR,  size=10), name="Static",  hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color=DYNAMIC_COLOR, size=10), name="Dynamic", hoverinfo="skip"))
    for gran, sym in gran_markers.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(color="#666666", size=10, symbol=sym),
            name=gran_display[gran], hoverinfo="skip"))

    for r in static_rows:
        if r["mode"] == "baseline" or r[xkey] is None or r["avg"] is None:
            continue
        gran, _ = extract_parts(r["method"])
        if gran is None:
            continue
        fig.add_trace(go.Scatter(x=[r[xkey]], y=[r["avg"]], mode="markers",
            marker=dict(symbol=gran_markers.get(gran, "circle"), color=STATIC_COLOR,
                        size=10, line=dict(color="white", width=0.5)),
            showlegend=False,
            hovertemplate=f"{r['method']}<br>{xlabel}: %{{x:.3f}}<br>Accuracy: %{{y:.2f}}<extra></extra>"))

    for r in dynamic_rows:
        if r["timed_out"] or r[xkey] is None or r["avg"] is None:
            continue
        gran, scoring = extract_parts(r["method"])
        if gran is None:
            continue
        fig.add_trace(go.Scatter(x=[r[xkey]], y=[r["avg"]], mode="markers",
            marker=dict(symbol=gran_markers.get(gran, "circle"), color=DYNAMIC_COLOR,
                        size=10, line=dict(color="white", width=0.5)),
            showlegend=False,
            hovertemplate=f"{r['method']}<br>{xlabel}: %{{x:.3f}}<br>Accuracy: %{{y:.2f}}<extra></extra>"))
        r_s = next((s for s in static_rows if extract_parts(s["method"]) == (gran, scoring)), None)
        if r_s and r_s[xkey] and r_s["avg"]:
            fig.add_trace(go.Scatter(x=[r_s[xkey], r[xkey]], y=[r_s["avg"], r["avg"]],
                mode="lines", line=dict(color="#cccccc", width=1),
                showlegend=False, hoverinfo="skip"))

    for method_key, sym, label, anchor in [
        ("baseline",     "star",    "Full precision", "left"),
        ("snapkv_static","diamond", "SnapKV",         "left"),
    ]:
        ref = next((r for r in static_rows if r["method"] == method_key), None)
        if ref and ref[xkey] and ref["avg"]:
            fig.add_trace(go.Scatter(x=[ref[xkey]], y=[ref["avg"]], mode="markers",
                marker=dict(symbol=sym, color="#000000" if method_key=="baseline" else "#444444",
                            size=14 if sym=="star" else 11),
                showlegend=False,
                hovertemplate=f"{label}<br>{xlabel}: %{{x:.3f}}<br>Accuracy: %{{y:.2f}}<extra></extra>"))
            fig.add_annotation(x=ref[xkey], y=ref["avg"], text=label, showarrow=False,
                xanchor=anchor, yanchor="bottom" if method_key=="baseline" else "top",
                font=dict(size=11, color="#000000" if method_key=="baseline" else "#444444"))

    fig.update_xaxes(title=xlabel, showgrid=False, zeroline=False)
    fig.update_yaxes(title="Average LongBench accuracy", showgrid=True,
        gridcolor="#eeeeee", zeroline=False)
    base_layout(fig, "", width=760, height=460)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
        xanchor="center", x=0.5))
    return fig

def fig3_tflops_vs_accuracy(static_rows, dynamic_rows, outdir):
    fig = _scatter_plotly(static_rows, dynamic_rows, "tflops",
                          "Profiled TFLOPs (1st example)")
    fig.update_layout(title="Figure 3: Attention TFLOPs vs. Accuracy — Static and Dynamic Methods")
    _save(fig, outdir, "fig3_tflops_vs_accuracy")

def fig4_throughput_vs_accuracy(static_rows, dynamic_rows, outdir):
    fig = _scatter_plotly(static_rows, dynamic_rows, "throughput",
                          "Generation throughput (tok/s)")
    fig.update_layout(title="Figure 4: Throughput vs. Accuracy — Static and Dynamic Methods")
    _save(fig, outdir, "fig4_throughput_vs_accuracy")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Full results table (CSV + LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

METHOD_DISPLAY = {
    "baseline":                               "Full Precision",
    "baseline_clusterpath_static":            "Full KV (ClusterKV path)",
    "snapkv_static":                          "SnapKV",
    "quest_static":                           "Quest",
    "clusterattn_static":                     "ClusterAttn (snapkv)",
    "clusterattn_quest_bounds_static":        "ClusterAttn + Quest bounds",
    "clusterattn_snapkv_static":              "ClusterAttn + SnapKV",
    "clusterattn_h2o_static":               "ClusterAttn + H2O",
    "clusterattn_recon_static":               "ClusterAttn + Recon",
    "clusterattn_expected_attention_static":  "ClusterAttn + Exp. Attn",
    "clusterattn_random_static":              "ClusterAttn + Random",
    "pagekv_quest_bounds_static":             "PageKV + Quest bounds",
    "pagekv_snapkv_static":                   "PageKV + SnapKV",
    "pagekv_h2o_static":                    "PageKV + H2O",
    "pagekv_recon_static":                    "PageKV + Recon",
    "pagekv_expected_attention_static":       "PageKV + Exp. Attn",
    "pagekv_random_static":                   "PageKV + Random",
    "tokenkv_quest_bounds_static":            "TokenKV + Quest bounds",
    "tokenkv_snapkv_static":                  "TokenKV + SnapKV",
    "tokenkv_h2o_static":                   "TokenKV + H2O",
    "tokenkv_recon_static":                   "TokenKV + Recon",
    "tokenkv_expected_attention_static":      "TokenKV + Exp. Attn",
    "tokenkv_random_static":                  "TokenKV + Random",
    "clusterattn_quest_bounds_dynamic":       "ClusterAttn + Quest bounds (dyn)",
    "clusterattn_h2o_dynamic":              "ClusterAttn + H2O (dyn)",
    "clusterattn_expected_attention_dynamic": "ClusterAttn + Exp. Attn (dyn)",
    "clusterattn_random_dynamic":             "ClusterAttn + Random (dyn)",
    "pagekv_quest_bounds_dynamic":            "PageKV + Quest bounds (dyn)",
    "pagekv_h2o_dynamic":                   "PageKV + H2O (dyn)",
    "pagekv_expected_attention_dynamic":      "PageKV + Exp. Attn (dyn)",
    "pagekv_random_dynamic":                  "PageKV + Random (dyn)",
    "tokenkv_quest_bounds_dynamic":           "TokenKV + Quest bounds (dyn)",
    "tokenkv_h2o_dynamic":                  "TokenKV + H2O (dyn)",
    "tokenkv_expected_attention_dynamic":     "TokenKV + Exp. Attn (dyn)",
    "tokenkv_random_dynamic":                 "TokenKV + Random (dyn)",
}

TABLE_ORDER = [
    "baseline", "baseline_clusterpath_static", "snapkv_static", "quest_static",
    "clusterattn_static",
    "clusterattn_quest_bounds_static", "clusterattn_snapkv_static",
    "clusterattn_h2o_static", "clusterattn_recon_static",
    "clusterattn_expected_attention_static", "clusterattn_random_static",
    "pagekv_quest_bounds_static", "pagekv_snapkv_static",
    "pagekv_h2o_static", "pagekv_recon_static",
    "pagekv_expected_attention_static", "pagekv_random_static",
    "tokenkv_quest_bounds_static", "tokenkv_snapkv_static",
    "tokenkv_h2o_static", "tokenkv_recon_static",
    "tokenkv_expected_attention_static", "tokenkv_random_static",
    "clusterattn_quest_bounds_dynamic", "clusterattn_h2o_dynamic",
    "clusterattn_expected_attention_dynamic", "clusterattn_random_dynamic",
    "pagekv_quest_bounds_dynamic", "pagekv_h2o_dynamic",
    "pagekv_expected_attention_dynamic", "pagekv_random_dynamic",
    "tokenkv_quest_bounds_dynamic", "tokenkv_h2o_dynamic",
    "tokenkv_expected_attention_dynamic", "tokenkv_random_dynamic",
]

SECTION_HEADERS = {
    "baseline":
        r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textit{Baselines}} \\",
    "clusterattn_static":
        r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textit{Block-density granularity (static)}} \\",
    "pagekv_quest_bounds_static":
        r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textit{Page-level granularity (static)}} \\",
    "tokenkv_quest_bounds_static":
        r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textit{Token-level granularity (static)}} \\",
    "clusterattn_quest_bounds_dynamic":
        r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textit{Dynamic methods}} \\",
}

def fig5_full_table(static_rows, dynamic_rows, outdir):
    all_rows = {r["method"]: r for r in static_rows}
    all_rows.update({r["method"]: r for r in dynamic_rows})

    csv_path = os.path.join(outdir, "fig5_full_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method","Mode","gov_report","hotpotqa","lcc","qasper",
                          "Average","Avg Latency (s)","Throughput (tok/s)","TFLOPs"])
        for key in TABLE_ORDER:
            r = all_rows.get(key)
            if r is None:
                continue
            timed = r.get("timed_out", False)
            writer.writerow([
                METHOD_DISPLAY.get(key, key), r["mode"],
                fmt(r["gov_report"])    if not timed else "timed out",
                fmt(r["hotpotqa"])      if not timed else "timed out",
                fmt(r["lcc"])           if not timed else "timed out",
                fmt(r["qasper"])        if not timed else "timed out",
                fmt(r["avg"])           if not timed else "timed out",
                fmt(r["latency"],   2)  if not timed else "timed out",
                fmt(r["throughput"],1)  if not timed else "timed out",
                fmt(r["tflops"],    1)  if not timed else "timed out",
            ])
    print(f"Saved {csv_path}")

    tex_path = os.path.join(outdir, "fig5_full_results.tex")
    compressed = [r for r in static_rows if r["mode"]=="static" and r["avg"] is not None]
    best_avg   = max(r["avg"] for r in compressed) if compressed else None

    def bold_if_best(val, best, decimals=2):
        if val is None: return r"\textemdash"
        s = f"{val:.{decimals}f}"
        return r"\textbf{" + s + "}" if (best and abs(val-best) < 0.005) else s

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Full results across all configurations on LongBench. "
        r"Accuracy metrics: F1 for \texttt{hotpotqa} and \texttt{qasper}, "
        r"ROUGE-L for \texttt{gov\_report}, Edit Similarity for \texttt{lcc}. "
        r"Dynamic methods marked \dag{} timed out; scores omitted. "
        r"All compressed methods use a token budget of 4{,}096. "
        r"Bold = best average accuracy among static compressed methods.}",
        r"\label{tab:full_results}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Method & Mode & gov\_report & hotpotqa & lcc & qasper & Avg & Lat (s) & Tput & TFLOPs \\",
        r"\midrule",
    ]
    nd = r"\textemdash"
    for key in TABLE_ORDER:
        r = all_rows.get(key)
        if r is None: continue
        if key in SECTION_HEADERS: lines.append(SECTION_HEADERS[key])
        timed   = r.get("timed_out", False)
        display = METHOD_DISPLAY.get(key, key) + (r" \dag{}" if timed else "")
        avg_str = bold_if_best(r["avg"], best_avg) if r["mode"]=="static" else fmt(r["avg"])
        if timed: avg_str = nd
        lines.append(" & ".join([
            display, r["mode"].capitalize(),
            fmt(r["gov_report"])    if not timed else nd,
            fmt(r["hotpotqa"])      if not timed else nd,
            fmt(r["lcc"])           if not timed else nd,
            fmt(r["qasper"])        if not timed else nd,
            avg_str,
            fmt(r["latency"],   2)  if not timed else nd,
            fmt(r["throughput"],1)  if not timed else nd,
            fmt(r["tflops"],    1)  if not timed else nd,
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Latency breakdown stacked bar
# ─────────────────────────────────────────────────────────────────────────────

def fig6_latency_breakdown(static_rows, dynamic_rows, outdir):
    dyn_scorings  = ["quest_bounds", "h2o", "random"]
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    static_prefill_color = "#8ecae6"
    static_decode_color = "#126782"
    static_line_color = "#023047"
    dynamic_prefill_color = "#f4a261"
    dynamic_decode_color = "#c1121f"
    dynamic_line_color = "#780000"
    entries = []
    for gran in granularities:
        for scoring in dyn_scorings:
            s = next((r for r in static_rows  if extract_parts(r["method"])==(gran,scoring)), None)
            d = next((r for r in dynamic_rows if extract_parts(r["method"])==(gran,scoring)), None)
            if not s: continue
            entries.append({
                "label":     f"{gran[:5]}\n{SCORING_LABELS[scoring]}",
                "gran":      gran,
                "scoring":   scoring,
                "s_lat":     s["latency"],
                "s_prefill": s["prefill_lat"],
                "s_decode":  s["decode_lat"],
                "d_prefill": d["prefill_lat"] if d and not d["timed_out"] else None,
                "d_decode":  d["decode_lat"]  if d and not d["timed_out"] else None,
                "timed_out": d["timed_out"]   if d else True,
            })

    x = np.arange(len(entries))
    bar_w, offset = 0.32, 0.18
    sx = [xi - offset for xi in x]
    dx = [xi + offset for xi in x]

    fig = go.Figure()
    static_prefill = [
        e["s_prefill"] if e["s_prefill"] is not None else None
        for e in entries
    ]
    static_decode = [
        e["s_decode"] if e["s_decode"] is not None else (
            e["s_lat"] if e["s_lat"] is not None else None
        )
        for e in entries
    ]
    fig.add_trace(go.Bar(x=sx, y=static_prefill, width=bar_w,
        marker=dict(color=static_prefill_color, opacity=0.72), name="Static — prefill",
        hovertemplate="Static prefill<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=sx, y=static_decode, width=bar_w,
        marker=dict(color=static_decode_color, opacity=0.92), name="Static — decode",
        text=[
            f"{((e['s_prefill'] or 0) + (e['s_decode'] or 0)):.1f}s"
            if e["s_prefill"] is not None and e["s_decode"] is not None
            else (f"{e['s_lat']:.1f}s" if e["s_lat"] is not None else "")
            for e in entries
        ],
        textposition="outside",
        hovertemplate="Static decode<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=dx, y=[e["d_prefill"] for e in entries], width=bar_w,
        marker=dict(color=dynamic_prefill_color, opacity=0.72), name="Dynamic — prefill",
        hovertemplate="Dynamic prefill<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=dx, y=[e["d_decode"] for e in entries], width=bar_w,
        marker=dict(color=dynamic_decode_color, opacity=0.92), name="Dynamic — decode",
        text=[f"{(e['d_prefill'] or 0)+(e['d_decode'] or 0):.1f}s"
              if e["d_prefill"] and e["d_decode"] else "" for e in entries],
        textposition="outside",
        hovertemplate="Dynamic decode<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=dx, y=[0.6 if e["timed_out"] else None for e in entries],
        width=bar_w, marker=dict(color=TIMEOUT_COLOR, opacity=0.6),
        name="Timed out",
        text=["T/O" if e["timed_out"] else "" for e in entries],
        textposition="outside", hoverinfo="skip"))

    for sep in [3, 6]:
        fig.add_vline(x=sep-0.5, line=dict(color="#cccccc", width=1, dash="dash"))
    for gi, gran in enumerate(granularities):
        fig.add_annotation(x=gi*3+1, y=1.08, xref="x", yref="paper",
            text=GRANULARITY_LABELS[gran].replace("\n"," "),
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=13, color="#222222"))

    fig.update_xaxes(tickmode="array", tickvals=x,
        ticktext=[e["label"] for e in entries], showgrid=False, zeroline=False)
    fig.update_yaxes(title="Average latency per example (s)",
        showgrid=True, gridcolor="#eeeeee", zeroline=False)
    base_layout(fig, "Figure 6: Latency Breakdown — Static vs. Dynamic",
                width=1020, height=500)
    fig.update_layout(barmode="stack",
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="center", x=0.5),
        margin=dict(l=60, r=25, t=90, b=90))
    _save(fig, outdir, "fig6_latency_breakdown")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Per-task heatmap — granularity × scoring backend
# ─────────────────────────────────────────────────────────────────────────────

def fig7_per_task_heatmap(static_rows, outdir):
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    scorings      = ["snapkv", "quest_bounds", "h2o", "recon", "expected_attention", "random"]
    tasks         = ["gov_report", "hotpotqa", "lcc", "qasper"]
    gran_labels   = {"clusterattn": "Block-density", "pagekv": "Page-level",
                     "tokenkv":     "Token-level"}
    scoring_short = {"snapkv": "SnapKV", "quest_bounds": "Quest", "h2o": "H2O",
                     "recon": "Recon", "expected_attention": "Exp.Attn", "random": "Random"}

    lookup = {}
    for r in static_rows:
        gran, scoring = extract_parts(r["method"])
        if gran and scoring:
            lookup[(gran, scoring)] = r

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=tasks,
        horizontal_spacing=0.16,
        vertical_spacing=0.24)

    for ti, task in enumerate(tasks):
        row_idx = ti // 2 + 1
        col_idx = ti % 2 + 1
        z, text = [], []
        for gran in granularities:
            row_z, row_t = [], []
            for scoring in scorings:
                r   = lookup.get((gran, scoring))
                val = r[task] if r else None
                row_z.append(val)
                row_t.append(f"{val:.1f}" if val is not None else "—")
            z.append(row_z)
            text.append(row_t)

        # Colour range tight around non-random values
        non_rand = [z[gi][si] for gi in range(len(granularities))
                    for si, s in enumerate(scorings)
                    if s != "random" and z[gi][si] is not None]
        zmin = min(non_rand) - 0.5 if non_rand else 25
        zmax = max(non_rand) + 0.5 if non_rand else 60

        fig.add_trace(go.Heatmap(
            z=z,
            x=[scoring_short[s] for s in scorings],
            y=[gran_labels[g]   for g in granularities],
            text=text, texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "#fff7bc"],
                [0.35, "#fec44f"],
                [0.7, "#d95f0e"],
                [1.0, "#7f2704"],
            ],
            zmin=zmin, zmax=zmax,
            showscale=(ti == 3),
            colorbar=dict(x=1.02, len=0.82,
                title=dict(text="Score", side="right"), thickness=14),
            hovertemplate=(
                "Granularity: %{y}<br>Scoring: %{x}<br>"
                f"Task: {task}<br>Score: %{{text}}<extra></extra>"),
        ), row=row_idx, col=col_idx)

    fig.update_layout(
        title=dict(
            text="Figure 7: Per-Task Accuracy Heatmap — Static Methods "
                 "(Granularity × Scoring Backend)",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=1000, height=720,
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=120, r=90, t=80, b=80),
    )
    _save(fig, outdir, "fig7_per_task_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Scoring backend comparison — grouped bars
# ─────────────────────────────────────────────────────────────────────────────

def fig8_scoring_backend_comparison(static_rows, outdir):
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    scorings      = ["snapkv", "quest_bounds", "h2o", "recon", "expected_attention", "random"]
    gran_colors   = {"clusterattn": "#333333", "pagekv": "#777777", "tokenkv": "#bbbbbb"}
    gran_labels   = {"clusterattn": "Block-density (clusterattn)",
                     "pagekv":      "Page-level (pagekv)",
                     "tokenkv":     "Token-level (tokenkv)"}

    lookup = {}
    for r in static_rows:
        gran, scoring = extract_parts(r["method"])
        if gran and scoring:
            lookup[(gran, scoring)] = r

    fig = go.Figure()
    for gran in granularities:
        x_vals, y_vals, hover = [], [], []
        for scoring in scorings:
            r   = lookup.get((gran, scoring))
            val = r["avg"] if r else None
            x_vals.append(SCORING_LABELS[scoring])
            y_vals.append(val)
            if r and val is not None:
                hover.append(
                    f"{gran_labels[gran]}<br>Scoring: {SCORING_LABELS[scoring]}<br>"
                    f"Avg: {val:.2f}<br>gov_report: {fmt(r['gov_report'])}<br>"
                    f"hotpotqa: {fmt(r['hotpotqa'])}<br>lcc: {fmt(r['lcc'])}<br>"
                    f"qasper: {fmt(r['qasper'])}")
            else:
                hover.append("No data")
        fig.add_trace(go.Bar(
            name=gran_labels[gran], x=x_vals, y=y_vals,
            marker=dict(color=gran_colors[gran], opacity=0.85),
            hovertext=hover, hoverinfo="text",
            text=[f"{v:.1f}" if v is not None else "" for v in y_vals],
            textposition="outside", textfont=dict(size=10),
        ))

    baseline = next((r for r in static_rows if r["method"] == "baseline"), None)
    snapkv   = next((r for r in static_rows if r["method"] == "snapkv_static"), None)
    if baseline and baseline["avg"]:
        fig.add_hline(y=baseline["avg"], line=dict(color="#888888", width=1, dash="dot"),
            annotation_text=f"Full precision ({baseline['avg']:.1f})",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#888888"))
    if snapkv and snapkv["avg"]:
        fig.add_hline(y=snapkv["avg"], line=dict(color="#555555", width=1, dash="dash"),
            annotation_text=f"SnapKV ({snapkv['avg']:.1f})",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#555555"))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Figure 8: Average Accuracy by Scoring Backend and Retrieval Granularity (Static)",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=920, height=480,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=80, t=70, b=60),
        yaxis=dict(title="Average LongBench accuracy", range=[28, 44],
            showgrid=True, gridcolor="#eeeeee", zeroline=False),
        xaxis=dict(title="Scoring backend", showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _save(fig, outdir, "fig8_scoring_backend_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Static-only latency comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig9_static_latency_comparison(static_rows, outdir):
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    grouped = {gran: [] for gran in granularities}
    for row in static_rows:
        gran, _ = extract_parts(row["method"])
        if gran in grouped and row.get("latency") is not None:
            grouped[gran].append(row)

    x_labels = [GRANULARITY_LINE_LABELS[gran] for gran in granularities]
    lows, q1s, medians, q3s, highs = [], [], [], [], []
    for gran in granularities:
        vals = sorted(row["latency"] for row in grouped[gran])
        if vals:
            lows.append(min(vals))
            q1s.append(float(np.percentile(vals, 25)))
            medians.append(float(np.percentile(vals, 50)))
            q3s.append(float(np.percentile(vals, 75)))
            highs.append(max(vals))
        else:
            lows.append(None)
            q1s.append(None)
            medians.append(None)
            q3s.append(None)
            highs.append(None)

    fig = go.Figure()

    for idx, gran in enumerate(granularities):
        color = GRANULARITY_COLORS[gran]
        fig.add_trace(go.Candlestick(
            x=[x_labels[idx]],
            open=[q1s[idx]],
            high=[highs[idx]],
            low=[lows[idx]],
            close=[q3s[idx]],
            increasing=dict(line=dict(color=color, width=2), fillcolor=color),
            decreasing=dict(line=dict(color=color, width=2), fillcolor=color),
            whiskerwidth=0.45,
            name=f"{GRANULARITY_LINE_LABELS[gran]} latency range",
            customdata=[medians[idx]],
            opacity=0.55,
            hovertemplate=(
                "%{x}<br>"
                "Min: %{low:.3f}s<br>"
                "Q1: %{open:.3f}s<br>"
                "Median: %{customdata:.3f}s<br>"
                "Q3: %{close:.3f}s<br>"
                "Max: %{high:.3f}s<extra></extra>"
            ),
            showlegend=False,
        ))

    for gran, label in zip(granularities, x_labels):
        rows = grouped[gran]
        if not rows:
            continue
        color = GRANULARITY_COLORS[gran]
        fig.add_trace(go.Scatter(
            x=[label] * len(rows),
            y=[row["latency"] for row in rows],
            mode="markers",
            marker=dict(size=9, color=color, opacity=0.85, line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=[
                f"{METHOD_DISPLAY.get(row['method'], row['method'])}<br>"
                f"Latency: {row['latency']:.3f}s<br>"
                f"Average accuracy: {fmt(row['avg'])}<extra></extra>"
                for row in rows
            ],
        ))

        best = min(rows, key=lambda row: row["latency"])
        fig.add_trace(go.Scatter(
            x=[label],
            y=[best["latency"]],
            mode="markers+text",
            marker=dict(symbol="star", size=17, color=color, line=dict(color="#111111", width=1.5)),
            text=[f"best: {METHOD_DISPLAY.get(best['method'], best['method'])}<br>{best['latency']:.2f}s"],
            textposition="bottom center",
            textfont=dict(size=10, color="#111111"),
            showlegend=False,
            hovertemplate=(
                f"Best latency in {label}<br>{best['method']}<br>"
                f"Latency: {best['latency']:.3f}s<br>"
                f"Average accuracy: {fmt(best['avg'])}<extra></extra>"
            ),
        ))

    baseline = next((r for r in static_rows if r["method"] == "baseline" and r.get("latency") is not None), None)
    if baseline:
        fig.add_hline(
            y=baseline["latency"],
            line=dict(color="#2b2b2b", width=2, dash="dash"),
            annotation_text=f"Baseline {baseline['latency']:.2f}s",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#2b2b2b"),
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="star", size=14, color="#555555", line=dict(color="#111111", width=1)),
        name="Best latency per granularity", hoverinfo="skip",
    ))

    y_values = [row["latency"] for rows in grouped.values() for row in rows]
    if baseline:
        y_values.append(baseline["latency"])
    y_max = max(y_values) * 1.15 if y_values else 10

    fig.update_layout(
        title=dict(
            text="Figure 9: Static Latency Distribution by Granularity",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=900, height=560,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=70, r=90, t=80, b=80),
        xaxis=dict(title="Granularity", showgrid=False, zeroline=False),
        yaxis=dict(title="Average latency per example (s)", range=[0, y_max],
                   showgrid=True, gridcolor="#eeeeee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
    )
    _save(fig, outdir, "fig9_static_latency_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Static-only throughput comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig10_static_throughput_comparison(static_rows, outdir):
    granularities = ["clusterattn", "pagekv", "tokenkv"]
    grouped = {gran: [] for gran in granularities}
    for r in static_rows:
        gran, _ = extract_parts(r["method"])
        if gran in grouped and r.get("throughput") is not None:
            grouped[gran].append(r)

    x_labels = [GRANULARITY_LINE_LABELS[gran] for gran in granularities]
    lows, q1s, medians, q3s, highs = [], [], [], [], []
    for gran in granularities:
        vals = sorted(row["throughput"] for row in grouped[gran])
        if vals:
            lows.append(min(vals))
            q1s.append(float(np.percentile(vals, 25)))
            medians.append(float(np.percentile(vals, 50)))
            q3s.append(float(np.percentile(vals, 75)))
            highs.append(max(vals))
        else:
            lows.append(None)
            q1s.append(None)
            medians.append(None)
            q3s.append(None)
            highs.append(None)

    fig = go.Figure()

    for idx, gran in enumerate(granularities):
        color = GRANULARITY_COLORS[gran]
        fig.add_trace(go.Candlestick(
            x=[x_labels[idx]],
            open=[q1s[idx]],
            high=[highs[idx]],
            low=[lows[idx]],
            close=[q3s[idx]],
            increasing=dict(line=dict(color=color, width=2), fillcolor=color),
            decreasing=dict(line=dict(color=color, width=2), fillcolor=color),
            whiskerwidth=0.45,
            opacity=0.55,
            name=f"{GRANULARITY_LINE_LABELS[gran]} throughput range",
            customdata=[medians[idx]],
            hovertemplate=(
                "%{x}<br>"
                "Min: %{low:.2f} tok/s<br>"
                "Q1: %{open:.2f} tok/s<br>"
                "Median: %{customdata:.2f} tok/s<br>"
                "Q3: %{close:.2f} tok/s<br>"
                "Max: %{high:.2f} tok/s<extra></extra>"
            ),
            showlegend=False,
        ))

    for gran, label in zip(granularities, x_labels):
        rows = grouped[gran]
        if not rows:
            continue
        color = GRANULARITY_COLORS[gran]
        fig.add_trace(go.Scatter(
            x=[label] * len(rows),
            y=[row["throughput"] for row in rows],
            mode="markers",
            marker=dict(size=9, color=color, opacity=0.85, line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=[
                f"{METHOD_DISPLAY.get(row['method'], row['method'])}<br>"
                f"Throughput: {row['throughput']:.2f} tok/s<br>"
                f"Average accuracy: {fmt(row['avg'])}<extra></extra>"
                for row in rows
            ],
        ))

        best = max(rows, key=lambda row: row["throughput"])
        fig.add_trace(go.Scatter(
            x=[label],
            y=[best["throughput"]],
            mode="markers+text",
            marker=dict(symbol="star", size=17, color=color, line=dict(color="#111111", width=1.5)),
            text=[f"best: {METHOD_DISPLAY.get(best['method'], best['method'])}<br>{best['throughput']:.1f} tok/s"],
            textposition="top center",
            textfont=dict(size=10, color="#111111"),
            showlegend=False,
            hovertemplate=(
                f"Best throughput in {label}<br>{best['method']}<br>"
                f"Throughput: {best['throughput']:.2f} tok/s<br>"
                f"Average accuracy: {fmt(best['avg'])}<extra></extra>"
            ),
        ))

    baseline = next((r for r in static_rows if r["method"] == "baseline" and r.get("throughput") is not None), None)
    if baseline:
        fig.add_hline(
            y=baseline["throughput"],
            line=dict(color="#2b2b2b", width=2, dash="dash"),
            annotation_text=f"Baseline {baseline['throughput']:.1f} tok/s",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#2b2b2b"),
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="star", size=14, color="#555555", line=dict(color="#111111", width=1)),
        name="Best throughput per granularity", hoverinfo="skip",
    ))

    y_values = [row["throughput"] for rows in grouped.values() for row in rows]
    if baseline:
        y_values.append(baseline["throughput"])
    y_max = max(y_values) * 1.15 if y_values else 20

    fig.update_layout(
        title=dict(
            text="Figure 10: Static Throughput Distribution by Granularity",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=900, height=560,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=70, r=90, t=80, b=80),
        xaxis=dict(title="Granularity", showgrid=False, zeroline=False),
        yaxis=dict(title="Throughput (tok/s)", range=[0, y_max],
                   showgrid=True, gridcolor="#eeeeee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
    )
    _save(fig, outdir, "fig10_static_throughput_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Static vs Dynamic average score comparison (Quest/H2O only)
# ─────────────────────────────────────────────────────────────────────────────

def fig11_avg_static_dynamic_comparison(static_rows, dynamic_rows, outdir):
    rows = _comparison_rows(static_rows, dynamic_rows, scorings=("quest_bounds", "h2o"))
    labels = [r["label"] for r in rows]
    static_vals = [r["static_avg"] for r in rows]
    dynamic_vals = [None if r["timed_out"] else r["dynamic_avg"] for r in rows]
    x = np.arange(len(rows))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=static_vals,
        mode="lines+markers+text",
        line=dict(color="#126782", width=3),
        marker=dict(color="#126782", size=9),
        name="Static",
        text=[f"{v:.2f}" if v is not None else "" for v in static_vals],
        textposition="top center",
        hovertemplate="Static<br>%{customdata}<br>Avg score: %{y:.2f}<extra></extra>",
        customdata=labels,
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=dynamic_vals,
        mode="lines+markers+text",
        line=dict(color="#c1121f", width=3),
        marker=dict(color="#c1121f", size=9),
        name="Dynamic",
        text=[f"{v:.2f}" if v is not None else "T/O" for v in dynamic_vals],
        textposition="bottom center",
        hovertemplate="Dynamic<br>%{customdata}<br>Avg score: %{y:.2f}<extra></extra>",
        customdata=labels,
    ))

    for idx, r in enumerate(rows):
        if r["timed_out"] or r["dynamic_avg"] is None or r["static_avg"] is None:
            continue
        delta = r["dynamic_avg"] - r["static_avg"]
        sign = "+" if delta >= 0 else ""
        fig.add_annotation(
            x=x[idx],
            y=max(r["static_avg"], r["dynamic_avg"]) + 0.55,
            text=f"{sign}{delta:.2f}",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=10, color="#555555"),
        )

    best_dynamic = max(
        (r for r in rows if not r["timed_out"] and r["dynamic_avg"] is not None),
        key=lambda r: r["dynamic_avg"],
        default=None,
    )
    if best_dynamic is not None:
        best_idx = rows.index(best_dynamic)
        fig.add_annotation(
            x=x[best_idx],
            y=best_dynamic["dynamic_avg"],
            text="best dynamic",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-35,
            font=dict(size=11, color="#222222"),
        )

    fig.update_layout(
        template="plotly_white",
        width=1020,
        height=500,
        title=dict(
            text="Figure 11: Average Score Trend — Static vs. Dynamic (Quest bounds and H2O)",
            x=0.5, xanchor="center"
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=70, r=50, t=70, b=120),
        xaxis=dict(
            title="Method combination",
            tickmode="array",
            tickvals=x,
            ticktext=labels,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Average score (higher is better)",
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _save(fig, outdir, "fig11_avg_static_dynamic_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Static vs Dynamic latency comparison (Quest/H2O only)
# ─────────────────────────────────────────────────────────────────────────────

def fig12_latency_static_dynamic_comparison(static_rows, dynamic_rows, outdir):
    rows = _comparison_rows(static_rows, dynamic_rows, scorings=("quest_bounds", "h2o"))
    labels = [r["label"] for r in rows]
    static_vals = [r["static_lat"] for r in rows]
    dynamic_vals = [None if r["timed_out"] else r["dynamic_lat"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=static_vals, y=labels, orientation="h",
        marker=dict(color=STATIC_COLOR, opacity=0.9),
        name="Static",
        text=[f"{v:.2f}s" if v is not None else "" for v in static_vals],
        textposition="outside",
        hovertemplate="Static<br>Avg latency: %{x:.3f}s<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=dynamic_vals, y=labels, orientation="h",
        marker=dict(color=DYNAMIC_COLOR, opacity=0.9),
        name="Dynamic",
        text=[f"{v:.2f}s" if v is not None else "T/O" for v in dynamic_vals],
        textposition="outside",
        hovertemplate="Dynamic<br>Avg latency: %{x:.3f}s<extra></extra>",
    ))

    for idx, r in enumerate(rows):
        if r["timed_out"] or r["dynamic_lat"] is None or r["static_lat"] is None:
            continue
        delta = r["dynamic_lat"] - r["static_lat"]
        sign = "+" if delta >= 0 else ""
        fig.add_annotation(
            x=max(r["static_lat"], r["dynamic_lat"]) + 0.65,
            y=labels[idx],
            text=f"{sign}{delta:.2f}s",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color="#555555"),
        )

    fig.add_annotation(
        x=2.403, y="tokenkv<br>Quest bounds",
        text="huge latency win",
        showarrow=True, arrowhead=2, ax=65, ay=-18,
        font=dict(size=11, color="#222222"),
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        width=980,
        height=520,
        title=dict(
            text="Figure 12: Latency Comparison — Static vs. Dynamic (Quest bounds and H2O)",
            x=0.5, xanchor="center"
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=170, r=100, t=70, b=60),
        xaxis=dict(title="Average latency per example (lower is better)", showgrid=True, gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="", autorange="reversed", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _save(fig, outdir, "fig12_latency_static_dynamic_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static")
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--outdir",  default="figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.static:
        static_rows = parse_static_csv(args.static)
        print(f"Loaded static rows from {args.static}")
    else:
        static_rows = parse_static_readme_table("README.md")
        print("Loaded static rows from README.md merged/current benchmark table")
    dynamic_rows = parse_dynamic_csv(args.dynamic)
    n_timed = sum(1 for r in dynamic_rows if r["timed_out"])
    print(f"Loaded {len(static_rows)} static, {len(dynamic_rows)} dynamic ({n_timed} timed out)")

    fig1_static_vs_dynamic(static_rows, dynamic_rows, args.outdir)
    fig2_static_accuracy_line(static_rows, args.outdir)
    fig3_tflops_vs_accuracy(static_rows, dynamic_rows, args.outdir)
    fig4_throughput_vs_accuracy(static_rows, dynamic_rows, args.outdir)
    fig5_full_table(static_rows, dynamic_rows, args.outdir)
    fig6_latency_breakdown(static_rows, dynamic_rows, args.outdir)
    fig7_per_task_heatmap(static_rows, args.outdir)
    fig8_scoring_backend_comparison(static_rows, args.outdir)
    fig9_static_latency_comparison(static_rows, args.outdir)
    fig10_static_throughput_comparison(static_rows, args.outdir)
    fig11_avg_static_dynamic_comparison(static_rows, dynamic_rows, args.outdir)
    fig12_latency_static_dynamic_comparison(static_rows, dynamic_rows, args.outdir)

    print("\nDone.")

if __name__ == "__main__":
    main()
