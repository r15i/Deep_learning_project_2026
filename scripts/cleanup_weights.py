import os
import glob
import re
import shutil
import sys

def cleanup_train(train_dir):
    pth_files = glob.glob(os.path.join(train_dir, "*.pth"))
    # Dictionary to keep track of the latest file for each (arch, task)
    latest_files = {}
    
    for pth in pth_files:
        basename = os.path.basename(pth)
        m_arch = re.search(r'_(resnet8|inception|transformer|microresnet)_', basename)
        m_task = re.search(r'_(activity|person_id)_', basename)
        m_time = re.search(r'^(\d{8}_\d{6})', basename)
        
        if m_arch and m_task and m_time:
            arch = m_arch.group(1)
            task = m_task.group(1)
            timestamp = m_time.group(1)
            key = (arch, task)
            if key not in latest_files or timestamp > latest_files[key][0]:
                latest_files[key] = (timestamp, pth)
                
    # Now delete all .pth files that are NOT in the latest_files list
    keep_paths = set(path for ts, path in latest_files.values())
    
    for pth in pth_files:
        if pth not in keep_paths:
            print(f"Deleting old weights: {pth}")
            try:
                os.remove(pth)
            except OSError as e:
                print(f"Error deleting {pth}: {e}")

def cleanup_test(test_dir):
    csv_files = sorted(glob.glob(os.path.join(test_dir, "test_results_*.csv")))
    if len(csv_files) > 1:
        for f in csv_files[:-1]:
            print(f"Deleting old CSV: {f}")
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error deleting {f}: {e}")
            
    graph_dirs = sorted(glob.glob(os.path.join(test_dir, "graphs_*")))
    if len(graph_dirs) > 1:
        for d in graph_dirs[:-1]:
            print(f"Deleting old graphs dir: {d}")
            try:
                shutil.rmtree(d)
            except OSError as e:
                print(f"Error deleting {d}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "weights/4nt0n"
        
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    if os.path.exists(train_dir):
        cleanup_train(train_dir)
    else:
        print(f"Warning: {train_dir} does not exist.")
        
    if os.path.exists(test_dir):
        cleanup_test(test_dir)
    else:
        print(f"Warning: {test_dir} does not exist.")
