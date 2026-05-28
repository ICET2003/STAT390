"""Create a single PDF for the complete experiment log bundle submission."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stat390-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stat390-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPORTS_DIR = Path("reports")
RESULT_MATRIX_PATH = REPORTS_DIR / "experiment_result_matrix.csv"
PDF_PATH = REPORTS_DIR / "complete_experiment_log_bundle_report.pdf"


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    lines = []
    for line in body.splitlines():
        if not line.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(line, width=118, replace_whitespace=False))

    page_lines = 38
    for start in range(0, len(lines), page_lines):
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        page_title = title if start == 0 else f"{title} continued"
        fig.text(0.055, 0.95, page_title, fontsize=17, fontweight="bold", va="top")
        y = 0.9
        for line in lines[start : start + page_lines]:
            fig.text(0.055, y, line, fontsize=8.5, family="monospace", va="top")
            y -= 0.022
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_metric_trajectory_page(pdf: PdfPages) -> None:
    results = pd.read_csv(RESULT_MATRIX_PATH)
    complete = results[results["status"].eq("complete")].copy()
    complete = complete.sort_values(["run_id", "model_index"])

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    for run_id, group in complete.groupby("run_id", sort=False):
        ax.plot(
            group["model_index"],
            group["primary_metric"],
            marker="o",
            linewidth=1.6,
            markersize=3,
            label=run_id,
        )

    ax.set_title("Metric Trajectory Plot", fontsize=17, fontweight="bold")
    ax.set_xlabel("Model index")
    ax.set_ylabel("Primary validation metric")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.text(
        0.08,
        0.045,
        "Classification uses weighted F1. Burnout-index regression uses R-squared.",
        fontsize=9,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sections = [
        ("Complete Experiment Log Bundle", REPORTS_DIR / "complete_experiment_log_bundle.md"),
        ("Keep / Discard / Crash Summary", REPORTS_DIR / "keep_discard_crash_summary.md"),
        ("Best Result vs. Baseline", REPORTS_DIR / "best_result_vs_baseline.md"),
        ("What Actually Worked Memo", REPORTS_DIR / "what_actually_worked_memo.md"),
    ]

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "Complete Experiment Log Bundle Report",
            "\n".join(
                [
                    "Included files:",
                    "- reports/complete_experiment_log_bundle.md",
                    "- reports/metric_trajectory_plot.svg",
                    "- reports/keep_discard_crash_summary.md",
                    "- reports/best_result_vs_baseline.md",
                    "- reports/what_actually_worked_memo.md",
                ]
            ),
        )
        for title, path in sections[:1]:
            add_text_page(pdf, title, path.read_text())
        add_metric_trajectory_page(pdf)
        for title, path in sections[1:]:
            add_text_page(pdf, title, path.read_text())

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
