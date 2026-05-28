"""Create a PDF for project-planning submission artifacts."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stat390-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stat390-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PDF_PATH = Path("reports/project_planning_artifacts_report.pdf")


def add_markdown_text(pdf: PdfPages, title: str, text: str) -> None:
    lines = []
    for line in text.splitlines():
        if not line.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(line, width=112, replace_whitespace=False))

    for start in range(0, len(lines), 38):
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        page_title = title if start == 0 else f"{title} continued"
        fig.text(0.055, 0.95, page_title, fontsize=17, fontweight="bold", va="top")
        y = 0.9
        for line in lines[start : start + 38]:
            fig.text(0.055, y, line, fontsize=8.5, family="monospace", va="top")
            y -= 0.022
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    sections = [
        ("Revised Project Statement and Agent Strategy", Path("docs/program.md")),
        ("Ablation or Comparison Table", Path("reports/ablation_comparison_table.md")),
        ("Final Two-Week Plan", Path("reports/final_two_week_plan.md")),
    ]
    with PdfPages(PDF_PATH) as pdf:
        add_markdown_text(
            pdf,
            "Project Planning Artifacts",
            "\n".join(
                [
                    "Included files:",
                    "- docs/program.md",
                    "- reports/ablation_comparison_table.md",
                    "- reports/final_two_week_plan.md",
                ]
            ),
        )
        for title, path in sections:
            add_markdown_text(pdf, title, path.read_text())

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
