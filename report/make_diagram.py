"""Draw the Q7d end-to-end process flow diagram.

    .venv/bin/python report/make_diagram.py      # writes report/figures/process_flow.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent / "figures" / "process_flow.png"

ML = "#dbe9f6"      # ML engineering steps
APP = "#e3f1df"     # application team steps
PROD = "#fdf0d5"    # shared production steps
DECISION = "#ffffff"
EDGE = "#3b4a5a"

fig, ax = plt.subplots(figsize=(11, 14.5))
ax.set_xlim(-3, 110)
ax.set_ylim(-16, 130)
ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.6",
                                fc=color, ec=EDGE, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, wrap=True)


def diamond(x, y, w, h, text):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=DECISION, ec=EDGE, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def arrow(x1, y1, x2, y2, label=None, label_pos=(0, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + label_pos[0], (y1 + y2) / 2 + label_pos[1], label,
                fontsize=9.5, color=EDGE, ha="center", va="center", style="italic")


# Lane titles and backgrounds
ax.add_patch(FancyBboxPatch((2, 50), 58, 76, boxstyle="round,pad=0.4", fc="#f5f9fd", ec="#b9cde3", lw=1))
ax.add_patch(FancyBboxPatch((66, 50), 42, 76, boxstyle="round,pad=0.4", fc="#f6fbf4", ec="#c3dcbc", lw=1))
ax.text(31, 124, "ML engineering", ha="center", fontsize=13, weight="bold", color="black")
ax.text(87, 124, "Application team", ha="center", fontsize=13, weight="bold", color="black")

# ML engineering lane
box(31, 116, 40, 5, "New customer data arrives", ML)
box(31, 106, 40, 5, "Register it as a new data asset version", ML)
box(31, 95, 44, 7, "Scheduled Azure ML pipeline\ndata prep → train / tune → evaluate", ML)
diamond(31, 82, 34, 12, "Beats the production\nmodel on held-out data?")
box(12, 68, 18, 7, "Keep current\nmodel and log\nthe run", ML)
box(38, 68, 26, 5, "Register new model version", ML)
box(38, 58, 26, 5, "Deploy to staging endpoint", ML)

arrow(31, 113.5, 31, 108.5)
arrow(31, 103.5, 31, 98.5)
arrow(31, 91.5, 31, 88)
arrow(18, 80, 13, 72, "no", (-2.5, 0))
arrow(38, 77.5, 38, 70.5, "yes", (2.5, 0))
arrow(38, 65.5, 38, 60.5)

# Application team lane
box(87, 116, 34, 7, "Agree on the API contract\n(OpenAPI / Swagger spec)", APP)
box(87, 101, 34, 9, "Build the agent screen\nand the dashboard against\nthe mock Flask API", APP)
box(87, 86, 34, 7, "Integration and contract\ntests against staging", APP)
arrow(87, 112.5, 87, 105.5)
arrow(87, 96.5, 87, 89.5)
arrow(51, 58, 70, 84, "staging ready", (6, -4))

# Shared production flow
box(55, 41, 56, 7, "Production endpoint: new version deployed as green\nwith 10 to 30% of traffic, blue keeps the rest", PROD)
arrow(87, 82.5, 76, 44.5, "release", (5, 0))
diamond(55, 28, 30, 10, "Live metrics OK?")
box(34, 16, 28, 5, "Shift 100% of traffic to green", PROD)
box(82, 16, 26, 5, "Roll back: green to 0%", PROD)
box(58, 5, 70, 5, "Production endpoint serves the applications", PROD)
box(33, -8.5, 50, 9, "Contact center: an incoming call looks up\nthe 10 fields by customer ID and shows the\nrisk and a suggested offer to the agent", APP)
box(86, -8.5, 30, 9, "Dashboard: nightly\nbatch scoring", APP)
arrow(40, 2.5, 36, -3.5)
arrow(76, 2.5, 82, -3.5)
arrow(55, 37.5, 55, 33)
arrow(43, 26, 37, 18.5, "yes", (-3, 1))
arrow(68, 26, 78, 18.5, "no", (3, 1))
arrow(34, 13.5, 42, 7.5)
arrow(82, 13.5, 70, 7.5)

# Monitoring loop back to the start
box(8, 34, 12, 20, "Monitoring\n\ndrift\nlatency\nerrors\nactual churn", "#f3e8f7")
arrow(23, 5, 8, 23.5)
ax.plot([1, 1], [44.5, 116], color=EDGE, lw=1.3)
ax.plot([1, 8], [44.5, 44.5], color=EDGE, lw=1.3)
arrow(1, 116, 11, 116)
ax.text(-0.5, 79, "drift or decay triggers retraining", rotation=90, fontsize=9.5,
        color=EDGE, ha="center", va="center", style="italic")

ax.set_title("End-to-end process: churn model in the contact center and dashboard",
             fontsize=14, weight="bold", pad=6)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}")
