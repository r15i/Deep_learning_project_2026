#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quick_benchmark.py — Rapid comparative evaluation across architectures and tasks.

Runs fast training (e.g. 2 epochs) and evaluation across all specified architectures 
and tasks, then outputs a consolidated comparison table and CSV.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
import pandas as pd

TASKS = ["activity", "person_id"]
ARCHS = ["inception", "resnet8", "transformer"]

# Optimal default hyperparams per task
TASK_HYPERPARAMS = {
    "activity": {"lr": "0.01", "dropout": "0.5"},
    "person_id": {"lr": "0.001", "dropout": "0.1"},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run fast comparative benchmark across architectures and tasks."
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/data/doppler_traces",
        help="Path to dataset directory.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="weights/benchmark",
        help="Directory to store benchmark weights and results.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs per run.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader worker threads.",
    )
    p.add_argument(
        "--fullvram",
        action="store_true",
        help="Preload entire dataset into VRAM.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Tasks to benchmark.",
    )
    p.add_argument(
        "--archs",
        nargs="+",
        default=ARCHS,
        choices=ARCHS,
        help="Architectures to benchmark.",
    )
    return p.parse_args()


def run_cmd(cmd: list):
    print(f"\n[EXEC] {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"ERROR: Command failed with code {res.returncode}")


def main():
    args = parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    summary_results = []

    total_runs = len(args.tasks) * len(args.archs)
    run_idx = 0
    start_time = time.time()

    print("=" * 70)
    print(f"🚀 STARTING FAST BENCHMARK ({args.epochs} EPOCHS)")
    print(f"   Tasks: {', '.join(args.tasks)}")
    print(f"   Architectures: {', '.join(args.archs)}")
    print(f"   Total combinations: {total_runs}")
    print("=" * 70)

    for task in args.tasks:
        hp = TASK_HYPERPARAMS.get(task, {"lr": "0.01", "dropout": "0.5"})
        for arch in args.archs:
            run_idx += 1
            print("\n" + "─" * 70)
            print(f"[{run_idx}/{total_runs}] TASK: {task.upper()} | ARCH: {arch.upper()}")
            print("─" * 70)

            # 1. Train
            train_cmd = [
                sys.executable,
                "train.py",
                "--dataset-path",
                args.dataset_path,
                "--arch",
                arch,
                "--task",
                task,
                "--epochs",
                str(args.epochs),
                "--lr",
                hp["lr"],
                "--dropout",
                hp["dropout"],
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--output-dir",
                train_dir,
            ]
            if args.fullvram:
                train_cmd.append("--fullvram")

            run_cmd(train_cmd)

            # Find the generated checkpoint for this run
            pth_files = glob.glob(os.path.join(train_dir, f"*_{arch}_{task}_*.pth"))
            if not pth_files:
                print(f"Warning: No pth weights found for {arch} on {task}")
                continue
            latest_pth = sorted(pth_files, key=os.path.getmtime)[-1]

            # 2. Test
            test_cmd = [
                sys.executable,
                "test.py",
                "--dataset-path",
                args.dataset_path,
                "--weights",
                latest_pth,
                "--arch",
                arch,
                "--task",
                task,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--output-dir",
                test_dir,
            ]
            if args.fullvram:
                test_cmd.append("--fullvram")

            run_cmd(test_cmd)

            # Find the latest generated CSV result
            csv_files = glob.glob(os.path.join(test_dir, "test_results_*.csv"))
            if csv_files:
                latest_csv = sorted(csv_files, key=os.path.getmtime)[-1]
                df = pd.read_csv(latest_csv)
                if not df.empty:
                    row = df.iloc[-1].to_dict()
                    summary_results.append(row)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🏁 BENCHMARK COMPLETE IN {elapsed:.2f}s")
    print("=" * 70)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved consolidated summary CSV to: {summary_csv_path}\n")

        # Display clean formatted comparison table
        cols_to_show = ["Task", "Architecture", "Loss", "Accuracy", "Precision", "Recall", "F1"]
        existing_cols = [c for c in cols_to_show if c in summary_df.columns]

        print("📊 MODEL COMPARISON RESULTS:")
        print(summary_df[existing_cols].to_string(index=False))

        print("\n🏆 BEST ARCHITECTURE PER TASK (by Accuracy / F1):")
        for task in args.tasks:
            task_df = summary_df[summary_df["Task"] == task]
            if not task_df.empty:
                best_acc_row = task_df.loc[task_df["Accuracy"].idxmax()]
                best_f1_row = task_df.loc[task_df["F1"].idxmax()]
                print(f"\n  ► {task.upper()}:")
                print(f"      Best Accuracy: {best_acc_row['Architecture']} ({best_acc_row['Accuracy']:.4f})")
                print(f"      Best F1 Score: {best_f1_row['Architecture']} ({best_f1_row['F1']:.4f})")
    else:
        print("No test results collected.")


if __name__ == "__main__":
    main()
