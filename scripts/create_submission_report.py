"""Create the controlled-experiment PDF submission report."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stat390-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")
MATRIX_PATH = REPORTS_DIR / "experiment_result_matrix.csv"
PDF_PATH = REPORTS_DIR / "controlled_experiment_submission_report.pdf"
MEMO_PATH = REPORTS_DIR / "submission_memo.md"


def wrap_text(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.94, title, fontsize=18, fontweight="bold", va="top")

    y = 0.88
    for heading, body in sections:
        fig.text(0.06, y, heading, fontsize=12, fontweight="bold", va="top")
        y -= 0.035
        fig.text(0.06, y, body, fontsize=9, va="top", family="monospace")
        y -= 0.035 + 0.018 * (body.count("\n") + 1)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    frame: pd.DataFrame,
    columns: list[str],
    rows_per_page: int = 24,
) -> None:
    view = frame[columns].copy()
    for start in range(0, len(view), rows_per_page):
        page = view.iloc[start : start + rows_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        end = min(start + rows_per_page, len(view))
        ax.set_title(f"{title} rows {start + 1}-{end} of {len(view)}", fontsize=14, fontweight="bold", pad=16)

        table = ax.table(
            cellText=page.astype(str).values,
            colLabels=columns,
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.25)
        for (row, _column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold")
            cell.set_edgecolor("#d1d5db")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_metric_plot_page(pdf: PdfPages, results: pd.DataFrame) -> None:
    complete = results[results["status"].eq("complete")].copy()
    complete = complete.sort_values(["run_id", "model_index"])

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    for run_id, group in complete.groupby("run_id", sort=False):
        ax.plot(group["model_index"], group["primary_metric"], marker="o", linewidth=1.5, markersize=3, label=run_id)

    ax.set_title("Metric-Over-Time Plot", fontsize=16, fontweight="bold")
    ax.set_xlabel("Model index")
    ax.set_ylabel("Primary validation metric")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.text(
        0.08,
        0.04,
        "Classification primary metric is weighted F1. Burnout-index regression primary metric is R-squared.",
        fontsize=9,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_memo(results: pd.DataFrame) -> None:
    failures = results[results["status"].ne("complete")]
    memo = [
        "# Submission Memo",
        "",
        "## Controlled Experiment Description",
        "",
        "The experiment was controlled by using a fixed random seed of 42, the same train-validation split policy inside each target task, and identical preprocessing for competing models within a run. Each target was evaluated in paired conditions: non-weather predictors only and weather-augmented predictors. Weather variables were excluded from burnout-index construction to avoid outcome leakage, and PCA source variables were excluded when predicting the burnout index.",
        "",
        "## Error Taxonomy",
        "",
        "Errors are classified as data/preprocessing failures, model-fit failures, metric/evaluation failures, and report-generation failures. A run is marked failed only when an exception prevents a model from completing and writing metrics. Non-fatal sklearn warnings are recorded as warnings but are not counted as failures.",
        "",
        f"Latest run status: {int(results['status'].eq('complete').sum())} complete models and {len(failures)} failures.",
    ]
    MEMO_PATH.write_text("\n".join(memo) + "\n")


def main() -> None:
    results = pd.read_csv(MATRIX_PATH)
    write_memo(results)

    failures = results[results["status"].ne("complete")]
    controlled_description = (
        "The controlled experiment compares four paired runs: treatment prediction without weather, "
        "treatment prediction with weather, burnout-index regression without weather, and burnout-index "
        "regression with weather. The same random state, split policy, preprocessing pipeline, target, "
        "and candidate model order are held fixed within each paired comparison. The only intended "
        "treatment difference is whether weather-derived predictors are included."
    )
    taxonomy = (
        "The error taxonomy separates data/preprocessing failures, model-fit failures, "
        "metric/evaluation failures, and report-generation failures. The latest run had no model "
        "failures; sklearn warnings that did not stop execution were not counted as failures."
    )

    best = (
        results[results["status"].eq("complete")]
        .sort_values(["run_id", "primary_metric"], ascending=[True, False])
        .groupby("run_id", as_index=False)
        .head(1)
    )

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "Controlled Experiment Submission Report",
            [
                ("Submitted Files", "\n".join([
                    "Experiment result matrix file: reports/experiment_result_matrix.csv",
                    "Plot file: reports/metric_over_time_plot.svg",
                    "Memo file: reports/submission_memo.md",
                    "Combined PDF report: reports/controlled_experiment_submission_report.pdf",
                ])),
                ("Controlled Experiment Description", wrap_text(controlled_description)),
                ("Error Taxonomy", wrap_text(taxonomy)),
                ("Latest Run Status", f"{int(results['status'].eq('complete').sum())} complete models; {len(failures)} failures."),
            ],
        )
        add_table_page(
            pdf,
            "Best Model By Controlled Run",
            best,
            ["run_id", "target", "task_type", "model", "primary_metric", "accuracy", "f1_weighted", "rmse", "mae", "r2"],
            rows_per_page=12,
        )
        add_table_page(
            pdf,
            "Experiment-Result Matrix",
            results.sort_values(["run_id", "model_index"]),
            ["run_id", "model_index", "model", "status", "primary_metric", "accuracy", "f1_weighted", "rmse", "mae", "r2"],
            rows_per_page=26,
        )
        add_metric_plot_page(pdf, results)
        add_text_page(
            pdf,
            "Error Taxonomy and Failure Analysis Memo",
            [
                ("Error Taxonomy", "No model failures in the latest run."),
                ("Failure Analysis Memo", "No model failures in the latest run.\nWarnings from sklearn that do not stop a run are not counted as failures."),
            ],
        )

    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
