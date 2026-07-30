import os
import glob
import wandb
import pandas as pd
from dotenv import load_dotenv
import re

load_dotenv(".env")
api = wandb.Api()
runs = api.runs("risoebiso-universit-di-padova/NNDL_Doppler")

# Get local files
local_pths = glob.glob("weights/*/train/*.pth")

# Map of timestamp -> local path
local_timestamps = {}
for path in local_pths:
    basename = os.path.basename(path)
    # Extracts the YYYYMMDD_HHMMSS timestamp from the beginning of the filename
    match = re.search(r"^(2026\d{4}_\d{6})", basename)
    if match:
        local_timestamps[match.group(1)] = path

missing = []
found = []

for r in runs[:30]:
    name = r.name
    # Skip "Eval" runs since they don't produce new weights
    if "_Eval_" in name:
        continue
    
    # Try to extract timestamp from the wandb run name
    match = re.search(r"(2026\d{4}_\d{6})", name)
    if match:
        timestamp = match.group(1)
        if timestamp in local_timestamps:
            found.append(f"✓ Found: {name} -> {local_timestamps[timestamp]}")
        else:
            missing.append(f"❌ MISSING: {name} (timestamp {timestamp})")

print(f"--- Verification Results ---")
print(f"Found {len(found)} matching local weights.")
print(f"Missing {len(missing)} local weights.")
print("\nDetails:")
for m in missing:
    print(m)
print("-" * 20)
for f in found:
    print(f)
