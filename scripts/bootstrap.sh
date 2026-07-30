#!/bin/bash
set -e

echo "=== [1/2] Checking Dataset Prerequisites ==="
if [ ! -d "/app/dataset/data/doppler_traces/S1a" ]; then
    echo "Dataset missing! Bootstrapping download..."
    uvx gdown 1vGHldHZVb9hQ1_YfFqZHqF8Ty1DQZEea -O /app/dataset/doppler_traces.zip
    echo "Extracting dataset..."
    rm -rf /app/dataset/data/doppler_traces
    uv run python -c "import zipfile; z=zipfile.ZipFile('/app/dataset/doppler_traces.zip', 'r')
for f in z.infolist():
    try: z.extract(f, '/app/dataset/data')
    except FileExistsError: pass"
fi

if [ ! -f "/app/dataset/data/doppler_traces/S1a/files_train_antennas_E,L,W,R,J.txt" ]; then
    echo "Dataset not preprocessed! Cloning SHARP repository and generating train/test splits..."
    rm -rf /tmp/SHARP && git clone https://github.com/francescamen/SHARP.git /tmp/SHARP
    sed -i 's/import tensorflow as tf/# import tensorflow as tf/g' /tmp/SHARP/Python_code/dataset_utility.py
    
    echo "Running train dataset prep..."
    uv run python /tmp/SHARP/Python_code/CSI_doppler_create_dataset_train.py /app/dataset/data/doppler_traces/ S1a,S1b,S1c,S2a,S2b,S3a,S4a,S4b,S5a 31 1 340 30 E,L,W,R,J 4
    
    echo "Running test dataset prep..."
    uv run python /tmp/SHARP/Python_code/CSI_doppler_create_dataset_test.py /app/dataset/data/doppler_traces/ S6a,S6b,S7a 31 1 340 30 E,L,W,R,J 4
    
    echo "Dataset preprocessing complete and ready!"
else
    echo "Dataset is already fully preprocessed and ready!"
fi

echo "=== [2/2] Running Target: $@ ==="
# Execute the target command
"$@"

echo "=== Target execution finished. ==="
