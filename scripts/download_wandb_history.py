import wandb
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
api = wandb.Api()
runs = api.runs("risoebiso-universit-di-padova/NNDL_Doppler")

hyperstack_runs = [r for r in runs if "hyperstack" in r.name and "Eval" not in r.name and r.state == "finished"]

activity_runs = [r for r in hyperstack_runs if "activity" in r.name]
person_runs = [r for r in hyperstack_runs if "person" in r.name]

def plot_task(runs_list, task_name, output_file):
    plt.figure(figsize=(10, 6))
    for r in runs_list:
        if "resnet8" in r.name:
            arch = "ResNet8"
            color = "blue"
        elif "inception" in r.name:
            arch = "Inception"
            color = "green"
        else:
            arch = "Transformer"
            color = "red"
            
        history = r.history(keys=["train/Loss", "val/Loss"], samples=10000)
        
        if "train/Loss" in history.columns:
            train_data = history[["train/Loss"]].dropna()
            plt.plot(range(len(train_data)), train_data.values, linestyle='--', color=color, alpha=0.5, label=f"{arch} Train")
        
        if "val/Loss" in history.columns:
            val_data = history[["val/Loss"]].dropna()
            # Often val is evaluated less frequently. We can just plot against its own index.
            # To make it line up roughly, we assume 1 val point per epoch and 1 train point per epoch.
            plt.plot(range(len(val_data)), val_data.values, linestyle='-', color=color, linewidth=2, label=f"{arch} Val")

    plt.title(f'Training and Validation Loss - {task_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

plot_task(activity_runs, "Activity Recognition", "paper/figures/loss_activity_combined.png")
plot_task(person_runs, "Person Identification", "paper/figures/loss_person_combined.png")
