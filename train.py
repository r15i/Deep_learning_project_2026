#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train & validate a model on the SHARP Doppler dataset.

Usage examples
--------------
    # ResNet34 with default hyper-parameters
    python train.py --dataset-path /data/doppler_traces

    # Transformer, 50 epochs, custom learning rate
    python train.py --dataset-path /data/doppler_traces \\
                    --arch transformer --epochs 50 --lr 5e-4

    # Custom output path for weights directory
    python train.py --dataset-path /data/doppler_traces \
                    --output-dir weights
"""

import argparse
import datetime
import sys
import os
import torch
import torch.multiprocessing as mp

# Allow CUDA tensors to be shared across workers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch.nn as nn

# from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support

from core.evaluation import evaluate_network
from core.constants import ACTIVITY_MAP
from core.data import build_dataloaders
from core.factory import build_model

load_dotenv()


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a ResNet34 or Transformer on the SHARP Doppler dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Root directory of the doppler_traces folder "
        "(e.g. /content/dataset_sharp_local/doppler_traces/).",
    )
    p.add_argument(
        "--arch",
        type=str,
        default="resnet8",
        choices=["resnet8", "inception", "transformer"],
        help="Model architecture to train.",
    )
    p.add_argument(
        "--task",
        type=str,
        default="activity",
        choices=["activity", "person_id"],
        help="Classification task to run.",
    )
    p.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataloaders.",
    )
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for dataloader.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="weights/train",
        help="Directory to save trained model artifacts and tensorboard logs.",
    )
    p.add_argument(
        "--fast-dry-run",
        action="store_true",
        help="If set, runs 2 batches per epoch and skips final evaluation.",
    )
    p.add_argument(
        "--fullvram",
        action="store_true",
        help="If set, loads the entire dataset into VRAM to eliminate I/O bottlenecks.",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────
# Training & Validation Helpers
# ───────────────────────────────────────────────────────────────────────────

def train_epoch(model, dl_train, optimizer, criterion, scaler, device, args, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(tqdm(dl_train, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")):
        if args.fast_dry_run and step >= 2: break
        
        optimizer.zero_grad()
        
        if args.task == "person_id":
            anchor, positive, negative, labels = batch
            anchor = anchor.to(device).view(anchor.size(0), -1, anchor.size(-1))
            positive = positive.to(device).view(positive.size(0), -1, positive.size(-1))
            negative = negative.to(device).view(negative.size(0), -1, negative.size(-1))
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                out_a = torch.nn.functional.normalize(model(anchor), p=2, dim=1)
                out_p = torch.nn.functional.normalize(model(positive), p=2, dim=1)
                out_n = torch.nn.functional.normalize(model(negative), p=2, dim=1)
                loss = criterion(out_a, out_p, out_n)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            dist_ap = torch.nn.functional.pairwise_distance(out_a, out_p)
            dist_an = torch.nn.functional.pairwise_distance(out_a, out_n)
            correct_triplets = (dist_ap < dist_an).float()
            
            all_preds.append(correct_triplets.detach().cpu())
            all_labels.append(torch.ones_like(correct_triplets).cpu())
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            data = data.view(data.size(0), -1, data.size(-1))

            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            all_preds.append(outputs.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

        total_loss += loss.item()
        samples_seen = (epoch * len(dl_train) + step) * args.batch_size
        wandb.log({"batch_train/Live Loss": loss.item(), "samples_seen": samples_seen})

    avg_train_loss = total_loss / len(dl_train)
    preds_cat, labels_cat = torch.cat(all_preds), torch.cat(all_labels)
    
    if args.task == "person_id":
        train_acc = preds_cat.mean().item()
        train_f1 = train_acc
    else:
        train_acc = (preds_cat.argmax(1) == labels_cat).float().mean().item()
        _, _, train_f1, _ = precision_recall_fscore_support(
            labels_cat.numpy(), preds_cat.argmax(1).numpy(), average="weighted", zero_division=0
        )

    print(f"  Train loss: {avg_train_loss:.4f}  |  Train acc: {train_acc:.4f}  |  Train F1: {train_f1:.4f}")
    return avg_train_loss, train_acc, train_f1

def validate_epoch(model, dl_val, criterion, device, args, epoch):
    model.eval()
    val_total_loss = 0.0
    v_preds, v_labels = [], []
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dl_val, desc=f"Epoch {epoch + 1}/{args.epochs} [val]")):
            if args.fast_dry_run and step >= 2: break
                
            if args.task == "person_id":
                anchor, positive, negative, labels = batch
                anchor = anchor.to(device).view(anchor.size(0), -1, anchor.size(-1))
                positive = positive.to(device).view(positive.size(0), -1, positive.size(-1))
                negative = negative.to(device).view(negative.size(0), -1, negative.size(-1))
                
                with torch.cuda.amp.autocast():
                    out_a = torch.nn.functional.normalize(model(anchor), p=2, dim=1)
                    out_p = torch.nn.functional.normalize(model(positive), p=2, dim=1)
                    out_n = torch.nn.functional.normalize(model(negative), p=2, dim=1)
                    loss = criterion(out_a, out_p, out_n)
                    
                val_total_loss += loss.item()
                dist_ap = torch.nn.functional.pairwise_distance(out_a, out_p)
                dist_an = torch.nn.functional.pairwise_distance(out_a, out_n)
                correct_triplets = (dist_ap < dist_an).float()
                v_preds.append(correct_triplets.cpu())
                v_labels.append(torch.ones_like(correct_triplets).cpu())
            else:
                bx, by = batch
                bx = bx.to(device).view(bx.size(0), -1, bx.size(-1))
                by = by.to(device)
                with torch.cuda.amp.autocast():
                    out = model(bx)
                    loss = criterion(out, by)
                val_total_loss += loss.item()
                v_preds.append(out.float().cpu())
                v_labels.append(by.cpu())

    v_preds, v_labels = torch.cat(v_preds), torch.cat(v_labels)
    val_loss = val_total_loss / len(dl_val)
    
    if args.task == "person_id":
        val_acc = v_preds.mean().item()
        val_f1 = val_acc
    else:
        val_acc = (v_preds.argmax(1) == v_labels).float().mean().item()
        _, _, val_f1, _ = precision_recall_fscore_support(
            v_labels.numpy(), v_preds.argmax(1).numpy(), average="weighted", zero_division=0
        )

    print(f"  Val loss:   {val_loss:.4f}  |  Val acc:   {val_acc:.4f}  |  Val F1: {val_f1:.4f}")
    return val_loss, val_acc, val_f1

# ───────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ───────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.backends.cudnn.benchmark = True

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{timestamp}_{args.arch}_{args.task}_ep{args.epochs}_lr{args.lr}_do{args.dropout}_bs{args.batch_size}.pth"
    output_path = os.path.join(args.output_dir, filename)

    dl_train, dl_val = build_dataloaders(args.dataset_path, args.batch_size, args.num_workers, args.task, fullvram=args.fullvram)

    num_classes = 128 if args.task == 'person_id' else len(ACTIVITY_MAP)
    model = build_model(args.arch, num_classes, args.dropout, task=args.task).to(device)
    print(f"\n{args.arch.upper()} model instantiated and moved to {device}.\n")

    criterion = nn.TripletMarginLoss(margin=1.0, p=2) if args.task == "person_id" else nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    patience, patience_counter = 40, 0

    env_name = os.environ.get("ENV_NAME", "local")
    print("Connecting to W&B server...")
    wandb.init(project="NNDL_Doppler", name=f"{env_name}_{args.arch}_{args.task}_{timestamp}", config=vars(args))
    print("Successfully connected to W&B.")

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, dl_train, optimizer, criterion, scaler, device, args, epoch)
        val_loss, val_acc, val_f1 = validate_epoch(model, dl_val, criterion, device, args, epoch)

        wandb.log({
            "train/Loss": train_loss, "train/Accuracy": train_acc, "train/F1_Score": train_f1,
            "val/Loss": val_loss, "val/Accuracy": val_acc, "val/F1_Score": val_f1, "epoch": epoch,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss, 'hyperparameters': vars(args)
            }
            torch.save(checkpoint, output_path)
            print(f"  ✓ Saved best model → {output_path}")
        else:
            patience_counter += 1
            print(f"  ! Early stopping counter: {patience_counter}/{patience}")

        scheduler.step(val_loss)
        
        # Log current learning rate to W&B
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train/Learning_Rate": current_lr, "epoch": epoch})
        
        print("-" * 72)
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

    print("\nTraining complete.")
    wandb.finish()

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    if not getattr(args, "fast_dry_run", False):
        if args.task != 'person_id':
            evaluate_network(dl_train, model, "Training", criterion, optimizer, args.task)
            evaluate_network(dl_val, model, "Validation", criterion, optimizer, args.task)
        else:
            print("Final evaluation pass skipped for Triplet Loss models (handled by test.py).")


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
