#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test trained models on the SHARP Doppler test set.

Usage examples
--------------
    # Test all checkpoints in a directory
    python test.py --dataset-path dataset/data/doppler_traces --weights-dir weights

    # Test a specific checkpoint
    python test.py --dataset-path dataset/data/doppler_traces --weights weights/best_model.pth
"""

import argparse
import sys
import os
import torch
import torch.multiprocessing as mp

# Allow CUDA tensors to be shared across workers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import csv
import datetime
import glob
import re

# import numpy as np
# import torch.nn as nn
import pandas as pd

# from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

# from core.constants import ACTIVITY_MAP
from core.data import build_test_loader
from core.evaluation import evaluate_file

load_dotenv()


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test trained models on the SHARP Doppler test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Root directory of the doppler_traces folder.",
    )
    p.add_argument(
        "--weights-dir",
        type=str,
        default="weights/train",
        help="Directory containing the .pth weights files to evaluate.",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to a specific .pth file. If omitted, tests all .pth files in --weights-dir.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="weights/test",
        help="Directory to save the CSV test results.",
    )
    p.add_argument(
        "--arch",
        type=str,
        default="resnet8",
        choices=["resnet8", "inception", "transformer"],
        help="Default model architecture if it cannot be inferred from filename.",
    )
    p.add_argument(
        "--task",
        type=str,
        default="activity",
        choices=["activity", "person_id"],
        help="Classification task to run.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Default dropout rate if it cannot be inferred.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for dataloader.",
    )
    p.add_argument(
        "--fast-dry-run",
        action="store_true",
        help="If set, evaluates only 2 batches for a fast test.",
    )
    p.add_argument(
        "--create-graphs",
        action="store_true",
        help="If set, generates confusion matrices and comparison graphs.",
    )
    p.add_argument(
        "--fullvram",
        action="store_true",
        help="If set, loads the entire dataset into VRAM to eliminate I/O bottlenecks.",
    )
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="If set, only evaluates the most recently modified matching .pth file.",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def main():
    # Parse command line arguments using the defined argument parser
    args = parse_args()

    # Check if a specific weights file was provided via arguments
    if args.weights:
        # If so, create a list containing just that single file path
        pth_files = [args.weights]
    else:
        # Otherwise, search the specified weights directory for all files ending in .pth
        pth_files = glob.glob(os.path.join(args.weights_dir, "*.pth"))
        # Filter by arch and task so we don't evaluate unrelated models
        if args.latest_only:
            pth_files = [f for f in pth_files if f"_{args.arch}_" in f and f"_{args.task}_" in f]
            # Sort by modification time
            pth_files.sort(key=os.path.getmtime)
            if pth_files:
                pth_files = [pth_files[-1]]
        else:
            pth_files.sort()

    # Check if the generated list of .pth files is empty
    if not pth_files:
        # Print a warning message indicating there is nothing to test
        print("No .pth files found to test.")
        # Exit the function early
        return

    # Determine whether a GPU is available to accelerate computation, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Inform the user which device is being used for testing
    print(f"Testing on {device}.")

    # Initialize a list to hold the evaluation metrics for all tested models
    results = []
    # Generate a formatted timestamp string representing the current time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine the output directory
    out_dir = args.output_dir
    # Ensure the output directory exists, safely creating it if it doesn't
    os.makedirs(out_dir, exist_ok=True)
    # Construct the full path for the resulting CSV report file, appending the timestamp
    csv_path = os.path.join(out_dir, f"test_results_{timestamp}.csv")

    # Initialize W&B task for evaluation
    env_name = os.environ.get("ENV_NAME", "")
    run_prefix = f"{env_name}_" if env_name else ""
    
    if len(pth_files) == 1:
        # If testing a single file, make the wandb run name specific
        basename = os.path.basename(pth_files[0])
        safe_name = basename.replace(".pth", "")
        wandb_name = f"{run_prefix}Eval_{safe_name}"
    else:
        # If evaluating multiple, use Evaluation_All
        wandb_name = f"{run_prefix}Eval_All_{timestamp}"
        
    wandb.init(project="NNDL_Doppler", name=wandb_name)

    # Define the column names for the evaluation report
    cols = [
        "File",
        "Task",
        "Architecture",
        "Dropout",
        "Batch_Size",
        "Loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    # Initialize CSV with header
    # Open the target CSV file in write mode
    with open(csv_path, "w", newline="") as f:
        # Initialize a CSV DictWriter with the defined column names
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        # Write the header row to the CSV file
        writer.writeheader()

    # Iterate over each .pth file found. Only show outer progress bar if there's more than 1 model to evaluate.
    pth_iter = tqdm(pth_files, desc="Evaluating Models") if len(pth_files) > 1 else pth_files
    for pth in pth_iter:
        # Extract just the filename from the full path
        basename = os.path.basename(pth)

        # Attempt to read parameters directly from the checkpoint dictionary
        arch = args.arch
        dropout = args.dropout
        batch_size = args.batch_size
        task_name = args.task
        
        try:
            checkpoint = torch.load(pth, map_location=device)
            if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
                hp = checkpoint["hyperparameters"]
                arch = hp.get("arch", arch)
                dropout = hp.get("dropout", dropout)
                task_name = hp.get("task", task_name)
                # We purposefully DO NOT extract batch_size from the training checkpoint 
                # because test batch size is hardware-dependent, not model-dependent.
            else:
                # Fallback to regex parsing for older checkpoints without hyperparameter dictionaries
                m_arch = re.search(r"_(resnet8|inception|transformer)_", basename)
                if m_arch:
                    arch = m_arch.group(1)

                m_task = re.search(
                    r"_(activity|person_id)_", basename
                )
                if m_task:
                    task_name = m_task.group(1)

                m_do = re.search(r"_do([\d\.]+)", basename)
                if m_do:
                    dropout = float(m_do.group(1))
        except Exception as e:
            print(f"Error reading checkpoint {basename}: {e}")
            continue

        # Temporarily set args.task for evaluate_file
        args.task = task_name

        try:
            # Build the test dataloader using the inferred or default batch size
            dl_test = build_test_loader(
                args.dataset_path, batch_size, args.num_workers, task_name, fullvram=args.fullvram
            )
            # Run the evaluation logic for the current model file and obtain the performance metrics
            metrics = evaluate_file(pth, arch, dropout, dl_test, device, args)
        except Exception as e:
            print(f"Skipping {basename} due to error: {e}")
            continue

        if metrics is None:
            continue

        # Add the filename to the metrics dictionary for reporting
        metrics["File"] = basename
        # Add the task to the metrics dictionary
        metrics["Task"] = task_name
        # Add the architecture to the metrics dictionary
        metrics["Architecture"] = arch
        # Add the dropout rate to the metrics dictionary
        metrics["Dropout"] = dropout
        # Add the batch size to the metrics dictionary
        metrics["Batch_Size"] = batch_size
        # Append the complete dictionary of metrics for this model to the results list
        results.append(metrics)

        # Log the test metrics to W&B
        wandb.log(
            {
                f"{arch}_{task_name}/Loss": metrics.get("Loss", 0),
                f"{arch}_{task_name}/Accuracy": metrics.get("Accuracy", 0),
                f"{arch}_{task_name}/F1_Score": metrics.get("F1", 0),
                "dropout_percent": int(dropout * 100),
            }
        )

        # Incrementally save to CSV
        # Open the CSV file in append mode to progressively save results
        with open(csv_path, "a", newline="") as f:
            # Initialize the DictWriter
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            # Write the metrics dictionary as a new row in the CSV
            writer.writerow(metrics)

    # Print a success message indicating completion and the location of the CSV
    print(f"\nAll tests complete! Results saved to: {csv_path}")
    # Convert the list of results into a pandas DataFrame, ordering columns as defined
    df = pd.DataFrame(results)[cols]
    # Print the full DataFrame to the console without the row index
    print(df.to_string(index=False))

    # Log the pandas DataFrame as a table to W&B
    wandb.log({"Evaluation Results": wandb.Table(dataframe=df)})
    
    # Generate requested graphs locally
    if args.create_graphs and not df.empty:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            graphs_dir = os.path.join(out_dir, f"graphs_{timestamp}")
            os.makedirs(graphs_dir, exist_ok=True)
            
            # 1. Plot Heatmaps for Metrics (Rows: Task, Cols: Architecture)
            metrics_to_plot = ["Accuracy", "F1", "Precision", "Recall", "Loss"]
            for metric in metrics_to_plot:
                if metric in df.columns:
                    plt.figure(figsize=(10, 6))
                    agg = 'min' if metric == 'Loss' else 'max'
                    pivot = df.pivot_table(index="Task", columns="Architecture", values=metric, aggfunc=agg)
                    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="Blues" if metric != "Loss" else "Reds")
                    plt.title(f"Best {metric} per Task and Architecture")
                    plt.tight_layout()
                    plt.savefig(os.path.join(graphs_dir, f"matrix_{metric}.png"))
                    plt.close()
                    
            # 2. Plot Confusion Matrices for each evaluated combination
            for r in results:
                task = r["Task"]
                arch = r["Architecture"]
                filename = r["File"]
                y_true = r.get("y_true")
                y_pred = r.get("y_pred")
                labels = r.get("labels", [])
                
                if y_true is not None and y_pred is not None:
                    plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(y_true, y_pred)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f"Confusion Matrix\n{task} | {arch}\n{filename}")
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    plt.tight_layout()
                    
                    safe_filename = filename.replace(".pth", "")
                    plt.savefig(os.path.join(graphs_dir, f"cm_{task}_{arch}_{safe_filename}.png"))
                    plt.close()
            
            print(f"\nGraphs successfully saved to {graphs_dir}/")
        except ImportError:
            print("\nCould not generate graphs: missing matplotlib or seaborn.")

    # Close the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
