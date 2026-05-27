from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(r"c:\Users\topra\OneDrive\Documents\CBL Urban\sim_outputs\app_scenarios")
NO_APP = OUT / "unmanaged_evening_timeseries.csv"
SMART = OUT / "smart_price_aware_timeseries.csv"

def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    return df

no_app = load(NO_APP)
smart = load(SMART)
cap = float(no_app["Zone_Capacity_MW"].iloc[0])

hours = no_app.index.hour + no_app.index.minute / 60.0

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

# --- Panel 1: total load ---
ax = axes[0]
ax.fill_between(no_app.index, no_app["Total_Load_MW"], alpha=0.18, color="#dc2626")
ax.plot(no_app.index, no_app["Total_Load_MW"], color="#dc2626", lw=2.4, label="Without smart charging (evening peak)")
ax.fill_between(smart.index, smart["Total_Load_MW"], alpha=0.15, color="#059669")
ax.plot(smart.index, smart["Total_Load_MW"], color="#059669", lw=2.4, label="With smart charging app (price-optimized)")
ax.axhline(cap, color="#6b7280", ls="--", lw=1.5, label=f"Zone capacity ({cap:.0f} MW)")
peak_no = no_app["Total_Load_MW"].max()
peak_yes = smart["Total_Load_MW"].max()
ax.annotate(f"Peak {peak_no:.0f} MW", xy=(no_app["Total_Load_MW"].idxmax(), peak_no), xytext=(12, 18), textcoords="offset points", fontsize=9, color="#dc2626", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
ax.annotate(f"Peak {peak_yes:.0f} MW", xy=(smart["Total_Load_MW"].idxmax(), peak_yes), xytext=(-60, 18), textcoords="offset points", fontsize=9, color="#059669", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2))
ax.set_ylabel("Total grid load (MW)")
ax.set_title("Smart charging levels daily load — sharper peak vs smoother profile", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)

std_no = no_app["Total_Load_MW"].std()
std_yes = smart["Total_Load_MW"].std()
ax.text(0.99, 0.03, f"Load variability (std): {std_no:.1f} MW without app  ->  {std_yes:.1f} MW with app", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#d1d5db"))

# --- Panel 2: congestion / stress ---
ax2 = axes[1]
ax2.fill_between(no_app.index, no_app["Stress_Ratio"], alpha=0.18, color="#dc2626")
ax2.plot(no_app.index, no_app["Stress_Ratio"], color="#dc2626", lw=2.2, label="Without smart charging")
ax2.plot(smart.index, smart["Stress_Ratio"], color="#059669", lw=2.2, label="With smart charging")
ax2.axhline(1.0, color="#6b7280", ls="--", lw=1.5, label="Grid limit (stress = 1.0)")
overload_no = int(no_app["Bottleneck"].sum())
overload_yes = int(smart["Bottleneck"].sum())
ax2.set_ylabel("Grid stress ratio")
ax2.set_xlabel("Time of day")
ax2.set_title("Congestion stress is spread out instead of spiking above the limit", fontsize=12)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, ls="--", alpha=0.4)
ax2.text(0.99, 0.97, f"Overload intervals: {overload_no} without  vs  {overload_yes} with", transform=ax2.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="#fef2f2" if overload_no > overload_yes else "#ecfdf5", ec="#fecaca"))

fig.autofmt_xdate()
fig.tight_layout()
out = OUT / "graph_daily_load_leveled.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)