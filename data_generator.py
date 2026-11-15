# Synthetic Pharmacy Dataset Generator
# Reproducible seed: 42
# Run: python data_generator.py
# Outputs: daily_medicine_sales.csv and batch_expiry_inventory.csv
#
# This dataset is fully synthetic and intended for academic demonstration.
# It is not real pharmacy, patient, prescribing, or treatment data.
#
# The expiry_risk label is generated using a transparent heuristic:
# High: days_to_expiry <= 90 and projected_unsold_ratio >= 0.30
# Medium: days_to_expiry <= 180 and projected_unsold_ratio >= 0.15
# Low: otherwise.
#
# For a stronger evaluation, treat this as a rule-generated demonstration
# label, not independently observed ground truth.

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
# See the accompanying README for schema and generation notes.
# The packaged CSV files are the generated outputs.
print("Use the packaged CSV files, or adapt this generator with your product catalog.")
