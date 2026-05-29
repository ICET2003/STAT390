"""Create final presentation slides for the STAT 390 project."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stat390-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stat390-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
PDF_PATH = REPORTS_DIR / "final_presentation.pdf"
NOTES_PATH = REPORTS_DIR / "final_presentation_notes.md"

TITLE = "Weather, Mental-Health Treatment, and Burnout Prediction"
AUTHOR = "IceT Thaewanarumitkul"

BG = "#f8fafc"
INK = "#111827"
MUTED = "#475569"
BLUE = "#2563eb"
GREEN = "#16a34a"
ORANGE = "#ea580c"
GRAY = "#64748b"


def new_slide(title: str):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.055, 0.91, title, fontsize=26, weight="bold", color=INK, va="top")
    ax.plot([0.055, 0.945], [0.86, 0.86], color="#cbd5e1", lw=1)
    return fig, ax


def add_footer(ax, slide_number: int) -> None:
    ax.text(0.055, 0.04, AUTHOR, fontsize=9, color=MUTED)
    ax.text(0.945, 0.04, str(slide_number), fontsize=9, color=MUTED, ha="right")


def add_bullets(ax, bullets: list[str], x=0.08, y=0.75, size=18, gap=0.09) -> None:
    for i, bullet in enumerate(bullets):
        ax.text(x, y - i * gap, f"- {bullet}", fontsize=size, color=INK, va="top")


def save_slide(pdf: PdfPages, fig, ax, slide_number: int) -> None:
    add_footer(ax, slide_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = pd.read_csv(RESULTS_DIR / "latest_experiment_result_matrix.csv")
    importance = pd.read_csv(RESULTS_DIR / "variable_importance.csv")
    return latest, importance


def add_table(ax, data, columns, bbox, font_size=12) -> None:
    table = ax.table(
        cellText=data,
        colLabels=columns,
        bbox=bbox,
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("white")


def slide_title(pdf: PdfPages, n: int) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.06, 0.72, TITLE, fontsize=34, weight="bold", color=INK, va="top")
    ax.text(0.06, 0.55, AUTHOR, fontsize=20, color=MUTED)
    ax.text(0.06, 0.46, "STAT 390 Final Presentation", fontsize=18, color=MUTED)
    ax.text(
        0.06,
        0.27,
        "Question: do weather variables add predictive value beyond work, sleep, health, and demographic variables?",
        fontsize=18,
        color=INK,
    )
    save_slide(pdf, fig, ax, n)


def slide_research_question(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Research Question")
    add_bullets(
        ax,
        [
            "Prior research suggests weather can affect mood, depression, and performance.",
            "This project tests a narrower predictive question.",
            "Does weather improve model performance after stronger personal and workplace predictors are already included?",
            "Targets: sought_treatment and PCA-derived burnout_index.",
        ],
        size=18,
    )
    save_slide(pdf, fig, ax, n)


def slide_data_pipeline(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Data and Pipeline")
    add_bullets(
        ax,
        [
            "Mental-health survey: treatment target and workplace mental-health predictors.",
            "Sleep-health dataset: sleep, stress, work, and health variables.",
            "Weather data: temperature, wind, pressure, sunlight, humidity, and related features.",
            "Burnout index: PCA target built from non-weather stress/sleep/work indicators.",
            "Weather was excluded from index construction to avoid leakage.",
        ],
        size=17,
        gap=0.082,
    )
    save_slide(pdf, fig, ax, n)


def slide_experiment_design(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Controlled Experiment Design")
    add_bullets(
        ax,
        [
            "For each target, compare non-weather features vs. weather-augmented features.",
            "Same random seed, split policy, preprocessing, and model list inside each comparison.",
            "Classification metric: weighted F1.",
            "Regression metric: R-squared.",
            "Full run: 88 completed models, 0 failures.",
        ],
        size=17,
        gap=0.085,
    )
    save_slide(pdf, fig, ax, n)


def slide_baseline_vs_best(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Baseline vs. Best Model")
    columns = ["Outcome", "Baseline", "Best", "Gain"]
    rows = [
        ["Treatment, no weather", "0.3915", "0.7962", "+0.4047"],
        ["Treatment, weather", "0.3915", "0.7963", "+0.4048"],
        ["Burnout, no weather", "~0.0000", "0.7938", "+0.7939"],
        ["Burnout, weather", "~0.0000", "0.7958", "+0.7959"],
    ]
    add_table(ax, rows, columns, [0.08, 0.32, 0.84, 0.42], font_size=15)
    ax.text(
        0.08,
        0.22,
        "Main point: the models learned real predictive structure, even though weather added only a small extra gain.",
        fontsize=17,
        color=INK,
    )
    save_slide(pdf, fig, ax, n)


def slide_weather_ablation(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Weather vs. Non-Weather Results")
    chart_ax = fig.add_axes([0.12, 0.22, 0.76, 0.52])
    labels = ["Treatment\nWeighted F1", "Burnout\nR-squared"]
    no_weather = [0.7962, 0.7938]
    weather = [0.7963, 0.7958]
    x = range(len(labels))
    width = 0.34
    chart_ax.bar([i - width / 2 for i in x], no_weather, width, label="No weather", color=GRAY)
    chart_ax.bar([i + width / 2 for i in x], weather, width, label="Weather", color=BLUE)
    chart_ax.set_xticks(list(x), labels, fontsize=13)
    chart_ax.set_ylim(0.78, 0.802)
    chart_ax.set_ylabel("Primary metric")
    chart_ax.grid(axis="y", alpha=0.25)
    chart_ax.legend()
    for i, (a, b) in enumerate(zip(no_weather, weather, strict=True)):
        chart_ax.text(i - width / 2, a + 0.0005, f"{a:.4f}", ha="center", fontsize=11)
        chart_ax.text(i + width / 2, b + 0.0005, f"{b:.4f}", ha="center", fontsize=11)
    ax.text(0.12, 0.12, "Weather improved both controlled runs, but the gains were small.", fontsize=17, color=INK)
    save_slide(pdf, fig, ax, n)


def slide_treatment_importance(pdf: PdfPages, n: int, importance: pd.DataFrame) -> None:
    fig, ax = new_slide("Treatment Prediction: Weather Signal")
    group = (
        importance[importance["run_id"].eq("treatment_weather_augmented")]
        .sort_values("importance_mean", ascending=False)
        .head(8)
        .sort_values("importance_mean")
    )
    chart_ax = fig.add_axes([0.12, 0.17, 0.78, 0.58])
    colors = [ORANGE if feature in {"wind_gust", "pressure_hpa"} else BLUE for feature in group["feature"]]
    chart_ax.barh(group["feature"], group["importance_mean"], color=colors)
    chart_ax.set_xlabel("Permutation importance")
    chart_ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.1,
        0.78,
        "Weather did not change F1 much, but wind_gust was the third most important treatment feature.",
        fontsize=16,
        color=INK,
    )
    save_slide(pdf, fig, ax, n)


def slide_burnout_importance(pdf: PdfPages, n: int, importance: pd.DataFrame) -> None:
    fig, ax = new_slide("Burnout Index: Weather Is Weaker")
    group = (
        importance[importance["run_id"].eq("burnout_index_weather_augmented")]
        .sort_values("importance_mean", ascending=False)
        .head(8)
        .sort_values("importance_mean")
    )
    chart_ax = fig.add_axes([0.12, 0.17, 0.78, 0.58])
    colors = [ORANGE if feature == "room_temperature_celsius" else GREEN for feature in group["feature"]]
    chart_ax.barh(group["feature"], group["importance_mean"], color=colors)
    chart_ax.set_xlabel("Permutation importance")
    chart_ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.1,
        0.78,
        "Burnout prediction is dominated by sleep, occupation, and day-type variables.",
        fontsize=16,
        color=INK,
    )
    save_slide(pdf, fig, ax, n)


def slide_interpretation(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Interpretation")
    add_bullets(
        ax,
        [
            "The project does not disprove prior weather-and-mental-health research.",
            "It asks whether weather adds prediction after stronger variables are already included.",
            "Weather effects may be indirect through sleep, stress, work patterns, or treatment behavior.",
            "Treatment prediction showed the clearest weather-related importance signal.",
            "Burnout index prediction was mostly explained by direct sleep/work/health variables.",
        ],
        size=17,
        gap=0.082,
    )
    save_slide(pdf, fig, ax, n)


def slide_limitations(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Limitations and Next Steps")
    add_bullets(
        ax,
        [
            "Weather matching is coarse and may not represent each person's true exposure.",
            "Timing may matter: heat waves, recent sunlight, seasonal lag, or multi-day exposure.",
            "The burnout index is constructed from variables close to sleep and work stress.",
            "Future work: finer geographic weather, date-aligned exposure windows, and causal controls.",
        ],
        size=17,
        gap=0.09,
    )
    save_slide(pdf, fig, ax, n)


def slide_conclusion(pdf: PdfPages, n: int) -> None:
    fig, ax = new_slide("Final Conclusion")
    add_bullets(
        ax,
        [
            "Models improved strongly over dummy baselines.",
            "Weather added weak incremental predictive value overall.",
            "Treatment prediction had the clearest weather importance signal: wind_gust and pressure_hpa.",
            "Burnout prediction was driven more by sleep, occupation, day type, and health variables.",
            "Final claim: weather may matter indirectly, but it is not a major predictor in this dataset.",
        ],
        size=17,
        gap=0.083,
    )
    save_slide(pdf, fig, ax, n)


def write_notes() -> None:
    notes = """# Final Presentation Notes

Presenter: IceT Thaewanarumitkul

## Main Talk Track

1. I tested whether weather variables improve prediction of mental-health treatment and a PCA burnout index.
2. The controlled experiment compared non-weather and weather-augmented feature sets under the same split and model list.
3. The models clearly beat dummy baselines: treatment improved from 0.3915 to about 0.796 weighted F1, and burnout improved from about 0 to about 0.796 R-squared.
4. Weather improved the final metrics only slightly.
5. The treatment model showed the clearest weather signal: wind_gust was the third most important feature, and pressure_hpa also appeared in the top ten.
6. Burnout prediction was dominated by sleep, occupation, day type, and health variables.
7. The conclusion is cautious: weather may matter indirectly, but it was not a major predictor in this dataset.
"""
    NOTES_PATH.write_text(notes)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _latest, importance = load_results()
    with PdfPages(PDF_PATH) as pdf:
        slide_title(pdf, 1)
        slide_research_question(pdf, 2)
        slide_data_pipeline(pdf, 3)
        slide_experiment_design(pdf, 4)
        slide_baseline_vs_best(pdf, 5)
        slide_weather_ablation(pdf, 6)
        slide_treatment_importance(pdf, 7, importance)
        slide_burnout_importance(pdf, 8, importance)
        slide_interpretation(pdf, 9)
        slide_limitations(pdf, 10)
        slide_conclusion(pdf, 11)
    write_notes()
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {NOTES_PATH}")


if __name__ == "__main__":
    main()
