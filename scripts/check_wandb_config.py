import os
import wandb
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
api = wandb.Api()
runs = api.runs("risoebiso-universit-di-padova/NNDL_Doppler")

results = []
for r in runs[:50]:
    # W&B runs have config and summary
    row = {
        "Name": r.name,
        "Created": r.created_at,
        "State": r.state
    }
    
    # Try to extract hyperparams from config
    for k, v in r.config.items():
        if k in ["arch", "task", "dropout", "batch_size", "lr"]:
            row[k] = v
            
    # Try to extract best test accuracy from summary
    test_acc = None
    for k, v in r.summary.items():
        if "Accuracy" in k and not k.startswith("val/") and not k.startswith("train/"):
            # This is a test accuracy metric (e.g. resnet8_person_id/Accuracy)
            if test_acc is None or v > test_acc:
                test_acc = v
        if k == "val/Accuracy" and "Val_Accuracy" not in row:
            row["Val_Accuracy"] = v
            
    if test_acc is not None:
        row["Test_Accuracy"] = test_acc
        
    results.append(row)

df = pd.DataFrame(results)
print(df.to_string())
