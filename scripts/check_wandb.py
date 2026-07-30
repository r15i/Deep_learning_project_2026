import os
import wandb
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
api = wandb.Api()
runs = api.runs("risoebiso-universit-di-padova/NNDL_Doppler")

results = []
for r in runs[:30]:
    row = {"Run": r.name, "State": r.state, "Created": r.created_at}
    for k, v in r.summary.items():
        if "Accuracy" in k:
            row[k] = v
    results.append(row)

df = pd.DataFrame(results)
print(df.to_string())
