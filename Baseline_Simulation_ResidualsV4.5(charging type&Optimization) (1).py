"""
Legacy entry point for the residuals prototype.

Logic lives in sim/residuals.py (fleet size, capacity, and paths match the main pipeline).

Prefer:
  python scripts/run_residuals.py
"""

from sim.residuals import main

if __name__ == "__main__":
    main()
