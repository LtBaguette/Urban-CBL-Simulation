from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

# Simulation Variables
CONFIDENCE_INTERVAL = 0.95  
DEGREES_OF_FREEDOM = 6      
SIMULATION_REPEATS = 3      

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Set" / "Data_Set"
DS5_DIR = next(DATA_DIR.glob("Dataset 5*"))
DS6_DIR = next(DATA_DIR.glob("Dataset 6*"))

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
TENNET_CONGESTION_FILE = DS5_DIR / "tennetcongestie.csv"
CONGESTION_PC6_FILE = DS5_DIR / "congestie_pc6.csv"

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
FOCUS_ZONE = "Z2"



def load_zone2_statistics() -> tuple[pd.Series, pd.Series]:
    """Calculate the baseline mean and sample standard deviation per hour for Zone Z2."""
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()

    zone2 = load_df.loc[load_df["zone_id"] == FOCUS_ZONE].copy()
    if zone2.empty:
        raise ValueError(f"No records found for zone {FOCUS_ZONE}")

    zone2["hour"] = zone2["timestamp"].dt.hour
    
    # Calculate both mean and standard deviation grouped by hour
    grouped = zone2.groupby("hour")["demand_MW"]
    hourly_mean = grouped.mean().reindex(range(24)).interpolate()
    hourly_std = grouped.std().reindex(range(24)).interpolate()
    
    return hourly_mean, hourly_std


def main() -> None:
    # 1. Load the real metrics from your CSV
    hourly_mean, hourly_std = load_zone2_statistics()
    
    # 2. Calculate the t-distribution critical boundary for the confidence interval
    alpha = 1.0 - CONFIDENCE_INTERVAL
    lower_cutoff = alpha / 2
    upper_cutoff = 1 - (alpha / 2)
    
    # Determine the critical t-score boundaries
    t_min = t.ppf(lower_cutoff, df=DEGREES_OF_FREEDOM)
    t_max = t.ppf(upper_cutoff, df=DEGREES_OF_FREEDOM)
    
    print(f"=== Configured Boundaries ===")
    print(f"Target Confidence Interval: {CONFIDENCE_INTERVAL * 100}%")
    print(f"Degrees of Freedom: {DEGREES_OF_FREEDOM}")
    print(f"Allowed t-score range: [{t_min:.4f} to {t_max:.4f}]")
    print("=" * 40 + "\n")

    # Establish the figure canvas outside the loop
    plt.figure(figsize=(11, 5.5))

    # 3. Simulation Loop
    sim_count = 0
    while sim_count < SIMULATION_REPEATS:
        sim_count += 1
        print(f"--- RUNNING SIMULATION REPEAT #{sim_count} ---")
        
        simulated_hours = []
        simulated_values = []
        
        for hour in range(24):
            mu = hourly_mean[hour]
            sigma = hourly_std[hour]
            
            # Rejection sampling loop to strictly enforce the CI boundary
            while True:
                sample_t = t.rvs(df=DEGREES_OF_FREEDOM)
                if t_min <= sample_t <= t_max:
                    simulated_val = mu + (sample_t * sigma)
                    simulated_val = max(0.0, simulated_val)  # Enforce non-negative demand
                    break
            
            simulated_hours.append(hour)
            simulated_values.append(simulated_val)
            
            print(f"Hour {hour:02d}:00 -> Base Mean: {mu:7.3f} MW | Simulated: {simulated_val:7.3f} MW")
            
        print("-" * 40 + "\n")
        
        # Plot each simulation line onto the canvas
        plt.plot(simulated_hours, simulated_values, alpha=0.6, label=f"Sim Run {sim_count}")

    # 4. Generate and lock down the Visual Framework
    plt.plot(hourly_mean.index, hourly_mean.values, color="black", linewidth=2.5, linestyle="--", label="Historical Mean")
    
    # FIX: Lock graph structure parameters to remain static
    plt.xlim(0, 23)
    plt.ylim(bottom=0)  # Forces Y axis to explicitly start at 0
    plt.autoscale(enable=True, axis='x', tight=True) 

    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Grid Demand (MW)")
    plt.title(f"Zone {FOCUS_ZONE}: Student's t-Distributed Demand Simulations ({CONFIDENCE_INTERVAL*100}% CI Bound)")
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

