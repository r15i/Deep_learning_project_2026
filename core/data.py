import os
import re
import pandas as pd
from torch.utils.data import DataLoader
from core.dopplerdataset import DopplerDataset, load_files

def extract_person_id(filepath: str) -> int:
    """Extracts 0-indexed person ID from filepath containing /S1a/ etc."""
    # Matches 'S' followed by digits, optionally followed by a letter, surrounded by path separators
    # Example: /S1a/ -> 0, /S6b/ -> 5
    match = re.search(r'/S(\d+)[a-z]?/', filepath)
    if match:
        return int(match.group(1)) - 1
    # Fallback if pattern not found
    return 0

import numpy as np
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Wraps a base dataset and yields (anchor, positive, negative, label) for Triplet Loss.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Extract labels from the underlying DopplerDataset
        if hasattr(dataset, 'labels') and isinstance(dataset.labels, dict):
            self.labels = np.array(list(dataset.labels.values()))
        else:
            self.labels = np.array([y for _, y in dataset])
            
        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }
        
    def __getitem__(self, index):
        anchor_img, label = self.dataset[index]
        
        # Ensure label is an int for dictionary lookup, as underlying dataset might return a tensor
        label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
        
        # Positive sample (same label, different index)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label_val])
        positive_img, _ = self.dataset[positive_index]
        
        # Negative sample (different label)
        negative_label = np.random.choice(list(set(self.label_to_indices.keys()) - {label_val}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]
        
        return anchor_img, positive_img, negative_img, label
        
    def __len__(self):
        return len(self.dataset)

def build_dataloaders(dataset_path: str, batch_size: int, num_workers: int = 0, task: str = "activity", fullvram: bool = False):
    """Return (train_loader, val_loader) built from the doppler_traces dir."""

    list_dir = sorted(os.listdir(dataset_path))
    # Filter out hidden files like .DS_Store
    list_dir = [d for d in list_dir if not d.startswith('.')]

    if task == "person_id":
        # The user requested binary classification on Subjects 4 and 5 only.
        test_dirs = ["S4a", "S4b", "S5a"]
        
        # Load pre-split train and val files directly from S4 and S5
        train_files = load_files(dataset_path, test_dirs, "files_train_antennas_E,L,W,R,J.txt", 0, len(test_dirs))
        val_files = load_files(dataset_path, test_dirs, "files_val_antennas_E,L,W,R,J.txt", 0, len(test_dirs))
        
        # Map labels based on directory string: S4 (ID 3) maps to 0, S5 (ID 4) maps to 1
        train_labels = [0 if extract_person_id(f) == 3 else 1 for f in train_files]
        val_labels = [0 if extract_person_id(f) == 3 else 1 for f in val_files]
        
        
    else:
        # --- Training ---
        train_files = load_files(dataset_path, list_dir,
                                 "files_train_antennas_E,L,W,R,J.txt", 0, 9)
        if task == "activity":
            train_labels = load_files(dataset_path, list_dir,
                                      "labels_train_antennas_E,L,W,R,J.txt", 0, 9)
        else:
            raise ValueError(f"Unknown task: {task}")

    train_df = pd.DataFrame({"data": train_files, "label": train_labels})

    ds_train = DopplerDataset(
        train_df["data"], train_df["label"].tolist(),
        mean=0., std=0.1, scale_range=(0.8, 1.2), ratio=0.1, dim=1, fullvram=fullvram
    )
    
    if task == "person_id":
        ds_train = TripletDataset(ds_train)
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        import torch
        
        counts = Counter(train_df["label"].tolist())
        weights = {cls: 1.0 / count for cls, count in counts.items()}
        sample_weights = [weights[lbl] for lbl in train_df["label"].tolist()]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)
        
    # --- Validation ---
    if task == "person_id":
        pass # val_files and val_labels are already loaded above
    else:
        val_files = load_files(dataset_path, list_dir,
                               "files_val_antennas_E,L,W,R,J.txt", 0, 9)
        if task == "activity":
            val_labels = load_files(dataset_path, list_dir,
                                    "labels_val_antennas_E,L,W,R,J.txt", 0, 9)
            
    val_df = pd.DataFrame({"data": val_files, "label": val_labels})

    ds_val = DopplerDataset(
        val_df["data"], val_df["label"].tolist(),
        mean=0.0, std=0.0, scale_range=(1.0, 1.0), ratio=0.0, dim=1, fullvram=fullvram
    )
    
    if task == "person_id":
        ds_val = TripletDataset(ds_val)
        
    # Prevent CUDA multiprocessing slowdown: if fullvram is True, we must use num_workers=0
    # Using spawn with workers on CUDA tensors introduces massive IPC overhead and locking, 
    # making it 6x slower than just keeping it in the main process!
    if fullvram:
        num_workers = 0
        
    # --- DataLoaders ---
    if task == "person_id":
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, 
                              num_workers=num_workers, pin_memory=not fullvram, persistent_workers=(num_workers > 0))
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=not fullvram, persistent_workers=(num_workers > 0))

    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=not fullvram, persistent_workers=(num_workers > 0))

    print(f"[{task.upper()}] Train samples: {len(ds_train)}  |  Val samples: {len(ds_val)}")
    return dl_train, dl_val


def build_test_loader(dataset_path: str, batch_size: int, num_workers: int = 0, task: str = "activity", fullvram: bool = False) -> DataLoader:
    if task == "person_id":
        test_dirs = ["S4a", "S4b", "S5a"]
        t_traces = load_files(dataset_path, test_dirs, "files_test_antennas_E,L,W,R,J.txt", 0, len(test_dirs))
        t_labels = [0 if extract_person_id(f) == 3 else 1 for f in t_traces]
    else:
        test_dirs = ["S6a", "S6b", "S7a"]
        
        t_traces = load_files(dataset_path, test_dirs, "files_complete_antennas_E,L,W,R,J.txt", 0, len(test_dirs))
        
        if task == "activity":
            t_labels = load_files(dataset_path, test_dirs, "labels_complete_antennas_E,L,W,R,J.txt", 0, len(test_dirs))
        else:
            raise ValueError(f"Unknown task: {task}")

    test_df = pd.DataFrame({"data": t_traces, "label": t_labels})
    ds_test = DopplerDataset(
        test_df["data"], test_df["label"].tolist(),
        mean=0.0, std=0.0, scale_range=(1.0, 1.0), ratio=0.0, dim=1, fullvram=fullvram
    )
    
    if task == "person_id":
        ds_test = TripletDataset(ds_test)
        
    if fullvram:
        num_workers = 0
        
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, pin_memory=not fullvram)
    return dl_test
