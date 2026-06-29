"""Data I/O for SpectraHawk RF datasets."""
import os
import sys
import glob
import numpy as np
import pandas as pd

def load_dronerf(root: str, window_samples=200000):
    """Loads DroneRF dataset (Al-Sa'd et al.)
    
    Expects CSV files where filename is {BUI}_{segment}.csv
    BUI format:
      1st char: 0=Background, 1=Drone
      2nd/3rd chars: 00=Bg, 01=AR, 10=Bebop, 11=Phantom
      4th/5th chars: Mode
      6th char: Band (H or L)
      
    Segments each CSV into fixed windows of `window_samples`.
    
    Returns
    -------
    X_list : list of np.ndarray (complex RF data)
    y_binary : np.ndarray (0=Background, 1=Drone)
    y_type : np.ndarray (0=Bg, 1=AR, 2=Bebop, 3=Phantom)
    groups : np.ndarray (string or int IDs for the parent recording)
    fs : float (Mocked at 20MHz or 40MHz, usually unspecified but we assume 20Mhz for feature extraction)
    """
    csv_files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"No .csv files found in {root}.")
        sys.exit(0)
        
    # Group files by BUI first
    bui_dict = {}
    for f in csv_files:
        b = os.path.basename(f).split('_')[0]
        if b not in bui_dict:
            bui_dict[b] = []
        bui_dict[b].append(f)
        
    # Take at most 5 files per BUI to prevent OOM
    subsampled = []
    for b, files in bui_dict.items():
        subsampled.extend(files[:5])
        
    csv_files = subsampled
    
    X_list = []
    y_binary_list = []
    y_type_list = []
    groups_list = []
    
    # DroneRF was captured at high sample rates, we'll assume 20 MHz
    fs = 20e6
    
    # Mapping for types
    type_map = {'00': 0, '01': 1, '10': 2, '11': 3}
    
    for fpath in csv_files:
        basename = os.path.basename(fpath)
        parts = basename.split('_')
        if len(parts) < 2:
            continue
            
        bui = parts[0] # e.g. 10100H
        
        if len(bui) < 5:
            continue
            
        is_drone = int(bui[0])
        drone_type_str = bui[1:3]
        d_type = type_map.get(drone_type_str, 0)
        
        # We group by the base BUI (first 5 chars, stripping the H/L band)
        # This ensures that H and L recordings of the same flight are treated as one group.
        group_id = bui[:-1] if bui[-1] in ('H', 'L') else bui
        
        # Load CSV. DroneRF CSVs in the Kaggle dataset are typically a single row of 10M values.
        # pandas.read_csv tries to create 10M columns which causes massive memory/time overhead.
        # We load it as text and parse directly.
        with open(fpath, 'r') as f:
            line = f.read().strip()
        vals = np.array(line.split(','), dtype=np.float32)
        iq = vals + 1j * np.zeros_like(vals)
            
        # Segment into windows
        for i in range(0, len(iq) - window_samples + 1, window_samples):
            window = iq[i:i + window_samples]
            X_list.append(window)
            y_binary_list.append(is_drone)
            y_type_list.append(d_type)
            groups_list.append(group_id)
            
    return X_list, np.array(y_binary_list), np.array(y_type_list), np.array(groups_list), fs
