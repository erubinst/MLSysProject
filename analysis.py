"""
KV Cache Compression Results — Figure Generator
Produces:
    - Figure 1: Static vs. Dynamic grouped bar chart (accuracy)
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

    marker = "### Current benchmark snapshot"
    if marker not in text:
        raise ValueError(f"Could not find '{marker}' in {path}")

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
    "baseline", "snapkv_static", "quest_static",
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
                "d_prefill": d["prefill_lat"] if d and not d["timed_out"] else None,
                "d_decode":  d["decode_lat"]  if d and not d["timed_out"] else None,
                "timed_out": d["timed_out"]   if d else True,
            })

    x = np.arange(len(entries))
    bar_w, offset = 0.32, 0.18
    sx = [xi - offset for xi in x]
    dx = [xi + offset for xi in x]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sx, y=[e["s_lat"] for e in entries], width=bar_w,
        marker=dict(color=STATIC_COLOR, opacity=0.85), name="Static (total)",
        text=[f"{e['s_lat']:.1f}s" if e["s_lat"] else "" for e in entries],
        textposition="outside",
        hovertemplate="Static total<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=dx, y=[e["d_prefill"] for e in entries], width=bar_w,
        marker=dict(color=DYNAMIC_COLOR, opacity=0.65), name="Dynamic — prefill",
        hovertemplate="Dynamic prefill<br>%{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Bar(x=dx, y=[e["d_decode"] for e in entries], width=bar_w,
        marker=dict(color=DYNAMIC_COLOR, opacity=1.0), name="Dynamic — decode",
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
    base_layout(fig, "Figure 6: Latency Breakdown — Static (total) vs. Dynamic (prefill + decode)",
                width=980, height=470)
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

    fig = make_subplots(rows=1, cols=4,
        subplot_titles=tasks,
        horizontal_spacing=0.14)

    for ti, task in enumerate(tasks):
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
            colorscale=[[0.0, "#f7f7f7"], [0.5, "#aaaaaa"], [1.0, "#333333"]],
            zmin=zmin, zmax=zmax,
            showscale=(ti == 3),
            colorbar=dict(x=1.02, len=0.9,
                title=dict(text="Score", side="right"), thickness=14),
            hovertemplate=(
                "Granularity: %{y}<br>Scoring: %{x}<br>"
                f"Task: {task}<br>Score: %{{text}}<extra></extra>"),
        ), row=1, col=ti+1)

    fig.update_layout(
        title=dict(
            text="Figure 7: Per-Task Accuracy Heatmap — Static Methods "
                 "(Granularity × Scoring Backend)",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=1300, height=380,
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=120, r=90, t=70, b=60),
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
    bar_color = "#7a0000"

    rows = [r for r in static_rows if r["latency"] is not None]
    rows.sort(key=lambda r: (r["latency"], r["method"]), reverse=True)

    labels = [METHOD_DISPLAY.get(r["method"], r["method"]) for r in rows]
    hover = []
    latencies = []
    for r in rows:
        latencies.append(r["latency"])
        hover.append(
            f"{METHOD_DISPLAY.get(r['method'], r['method'])}<br>"
            f"Average latency: {r['latency']:.3f}s<br>"
            f"Average accuracy: {fmt(r['avg'])}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=latencies,
        y=labels,
        orientation="h",
        marker=dict(color=bar_color, opacity=0.9),
        text=[f"{lat:.2f}s" for lat in latencies],
        textposition="outside",
        textfont=dict(size=10),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text="Figure 9: Static-Only Average Latency Comparison",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=980, height=760,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=180, r=70, t=80, b=60),
        xaxis=dict(title="Average latency per example (s)", showgrid=True,
                   gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="Static methods", autorange="reversed", showgrid=False),
    )
    _save(fig, outdir, "fig9_static_latency_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Static-only throughput comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig10_static_throughput_comparison(static_rows, outdir):
    bar_color = "#7a0000"

    rows = [r for r in static_rows if r["throughput"] is not None]
    rows.sort(key=lambda r: (r["throughput"], r["method"]), reverse=True)

    labels = [METHOD_DISPLAY.get(r["method"], r["method"]) for r in rows]
    hover = []
    throughputs = []
    for r in rows:
        throughputs.append(r["throughput"])
        hover.append(
            f"{METHOD_DISPLAY.get(r['method'], r['method'])}<br>"
            f"Throughput: {r['throughput']:.2f} tok/s<br>"
            f"Average accuracy: {fmt(r['avg'])}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=throughputs,
        y=labels,
        orientation="h",
        marker=dict(color=bar_color, opacity=0.9),
        text=[f"{t:.1f} tok/s" for t in throughputs],
        textposition="outside",
        textfont=dict(size=10),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text="Figure 10: Static-Only Throughput Comparison",
            x=0.5, xanchor="center"),
        template="plotly_white",
        width=980, height=760,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=180, r=70, t=80, b=60),
        xaxis=dict(title="Throughput (tok/s)", showgrid=True,
                   gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="Static methods", autorange="reversed", showgrid=False),
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

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=static_vals, y=labels, orientation="h",
        marker=dict(color=STATIC_COLOR, opacity=0.9),
        name="Static",
        text=[f"{v:.2f}" if v is not None else "" for v in static_vals],
        textposition="outside",
        hovertemplate="Static<br>Avg score: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=dynamic_vals, y=labels, orientation="h",
        marker=dict(color=DYNAMIC_COLOR, opacity=0.9),
        name="Dynamic",
        text=[f"{v:.2f}" if v is not None else "T/O" for v in dynamic_vals],
        textposition="outside",
        hovertemplate="Dynamic<br>Avg score: %{x:.2f}<extra></extra>",
    ))

    for idx, r in enumerate(rows):
        if r["timed_out"] or r["dynamic_avg"] is None or r["static_avg"] is None:
            continue
        delta = r["dynamic_avg"] - r["static_avg"]
        sign = "+" if delta >= 0 else ""
        fig.add_annotation(
            x=max(r["static_avg"], r["dynamic_avg"]) + 0.7,
            y=labels[idx],
            text=f"{sign}{delta:.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color="#555555"),
        )

    fig.add_annotation(
        x=40.16, y="tokenkv<br>Quest bounds",
        text="best dynamic",
        showarrow=True, arrowhead=2, ax=55, ay=-12,
        font=dict(size=11, color="#222222"),
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        width=980,
        height=520,
        title=dict(
            text="Figure 11: Avg Performance — Static vs. Dynamic (Quest bounds and H2O)",
            x=0.5, xanchor="center"
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=170, r=90, t=70, b=60),
        xaxis=dict(title="Average score (higher is better)", showgrid=True, gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="", autorange="reversed", showgrid=False),
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
        print("Loaded static rows from README.md current benchmark table")
    dynamic_rows = parse_dynamic_csv(args.dynamic)
    n_timed = sum(1 for r in dynamic_rows if r["timed_out"])
    print(f"Loaded {len(static_rows)} static, {len(dynamic_rows)} dynamic ({n_timed} timed out)")

    fig1_static_vs_dynamic(static_rows, dynamic_rows, args.outdir)
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
