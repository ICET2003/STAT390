"""Create report-ready visualizations for the weather/burnout experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_MATRIX_PATH = Path("reports/experiment_result_matrix.csv")
SURVEY_WEATHER_PATH = Path("data/processed/survey_weather_merged.csv")
BURNOUT_INDEX_PATH = Path("data/processed/burnout_index.csv")
SLEEP_DATA_PATH = Path("data/raw/sleep_health_dataset.csv")
FIGURES_DIR = Path("figures")


def save_current_figure(path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_ablation_summary(results: pd.DataFrame) -> None:
    best = (
        results[results["status"].eq("complete")]
        .sort_values(["run_id", "primary_metric"], ascending=[True, False])
        .groupby("run_id")
        .head(1)
        .copy()
    )
    best["label"] = best["run_id"].map(
        {
            "treatment_non_weather": "Treatment\nNo weather",
            "treatment_weather_augmented": "Treatment\nWeather",
            "burnout_index_non_weather": "Burnout index\nNo weather",
            "burnout_index_weather_augmented": "Burnout index\nWeather",
        }
    )
    colors = ["#64748b", "#0ea5e9", "#64748b", "#0ea5e9"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(best["label"], best["primary_metric"], color=colors)
    plt.ylabel("Primary validation metric")
    plt.title("Best Model Performance: Non-Weather vs Weather-Augmented")
    plt.ylim(0, max(best["primary_metric"]) + 0.08)
    for bar, value in zip(bars, best["primary_metric"], strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            fontsize=9,
        )
    save_current_figure(FIGURES_DIR / "ablation_best_model_bar.png")


def plot_weather_delta() -> None:
    rows = pd.DataFrame(
        [
            {
                "target": "sought_treatment",
                "metric": "Weighted F1",
                "delta": 0.7963362560213959 - 0.796165854401462,
            },
            {
                "target": "burnout_index",
                "metric": "R-squared",
                "delta": 0.7958401623394091 - 0.7938052716439359,
            },
        ]
    )
    plt.figure(figsize=(7, 4))
    colors = ["#22c55e" if value > 0 else "#ef4444" for value in rows["delta"]]
    bars = plt.bar(rows["target"], rows["delta"], color=colors)
    plt.axhline(0, color="#111827", linewidth=1)
    plt.ylabel("Weather improvement over non-weather")
    plt.title("Incremental Value of Weather Features")
    for bar, value in zip(bars, rows["delta"], strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0001,
            f"+{value:.4f}",
            ha="center",
            fontsize=9,
        )
    save_current_figure(FIGURES_DIR / "weather_delta_bar.png")


def plot_model_rankings(results: pd.DataFrame) -> None:
    complete = results[results["status"].eq("complete")].copy()
    runs = [
        "treatment_non_weather",
        "treatment_weather_augmented",
        "burnout_index_non_weather",
        "burnout_index_weather_augmented",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for axis, run_id in zip(axes, runs, strict=True):
        run = (
            complete[complete["run_id"].eq(run_id)]
            .sort_values("primary_metric", ascending=False)
            .head(8)
            .sort_values("primary_metric")
        )
        axis.barh(run["model"], run["primary_metric"], color="#2563eb")
        axis.set_title(run_id.replace("_", " "))
        axis.set_xlabel("Primary metric")
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle("Top 8 Models by Run", fontsize=14, fontweight="bold")
    save_current_figure(FIGURES_DIR / "model_rankings_top8.png")


def plot_weather_treatment_relationship() -> None:
    df = pd.read_csv(SURVEY_WEATHER_PATH)
    if "temperature_f" not in df.columns or "sought_treatment" not in df.columns:
        return

    by_state = (
        df.groupby("state_code", as_index=False)
        .agg(
            treatment_rate=("sought_treatment", "mean"),
            temperature_f=("temperature_f", "mean"),
            humidity=("humidity", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .dropna(subset=["temperature_f", "treatment_rate"])
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        by_state["temperature_f"],
        by_state["treatment_rate"],
        s=55,
        color="#0ea5e9",
        edgecolor="#0f172a",
        linewidth=0.4,
    )
    for _, row in by_state.iterrows():
        plt.text(row["temperature_f"], row["treatment_rate"], row["state_code"], fontsize=7)
    plt.xlabel("State-capital temperature (F)")
    plt.ylabel("Treatment rate")
    plt.title("Weather and Mental-Health Treatment Rate by State")
    save_current_figure(FIGURES_DIR / "temperature_vs_treatment_rate.png")


def plot_burnout_weather_relationship() -> None:
    sleep = pd.read_csv(SLEEP_DATA_PATH)
    burnout = pd.read_csv(BURNOUT_INDEX_PATH)
    df = sleep.merge(burnout, on="person_id", how="inner")
    if "room_temperature_celsius" not in df.columns:
        return

    sample = df.sample(n=min(6000, len(df)), random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(
        sample["room_temperature_celsius"],
        sample["burnout_index"],
        s=8,
        alpha=0.22,
        color="#7c3aed",
    )
    grouped = (
        df.assign(temp_bin=pd.cut(df["room_temperature_celsius"], bins=20))
        .groupby("temp_bin", observed=True)
        .agg(
            temp_mid=("room_temperature_celsius", "mean"),
            burnout_mean=("burnout_index", "mean"),
        )
        .dropna()
    )
    plt.plot(grouped["temp_mid"], grouped["burnout_mean"], color="#111827", linewidth=2)
    plt.xlabel("Room temperature (C)")
    plt.ylabel("PCA burnout index")
    plt.title("Room Temperature vs. Burnout Index")
    save_current_figure(FIGURES_DIR / "room_temperature_vs_burnout_index.png")


def main() -> None:
    results = pd.read_csv(RESULT_MATRIX_PATH)
    plot_ablation_summary(results)
    plot_weather_delta()
    plot_model_rankings(results)
    plot_weather_treatment_relationship()
    plot_burnout_weather_relationship()
    print(f"Wrote visualizations to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
