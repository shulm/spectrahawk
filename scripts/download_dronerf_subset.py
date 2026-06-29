import os
import zipfile
import kaggle

def download_subset():
    kaggle.api.authenticate()
    dataset = 'alishawang/dronerf'
    
    print("Fetching file list (all files)...")
    res = kaggle.api.dataset_list_files(dataset, page_size=1000)
    files = res.files if hasattr(res, 'files') else []
    if not files and hasattr(res, 'get'):
        files = res.get('files', [])
    # Sometimes it returns a DatasetFiles object
    if not isinstance(files, list):
        files = list(files)
        
    print(f"Found {len(files)} files in dataset.")
    
    downloads = {}
    
    for f in files:
        name = getattr(f, 'name', f)
        if not str(name).endswith('.csv'):
            continue
        
        basename = os.path.basename(str(name))
        if 'H' not in basename:
            continue
            
        parts = basename.split('_')
        if len(parts) != 2:
            continue
            
        bui = parts[0]
        
        if bui not in downloads:
            downloads[bui] = []
            
        # Cap at 5 segments per BUI so we get enough background files
        if len(downloads[bui]) < 5:
            downloads[bui].append(str(name))
            
    to_download = []
    for bui, flist in downloads.items():
        to_download.extend(flist)
        
    print(f"Selected {len(to_download)} files across {len(downloads)} recordings.")
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dronerf')
    os.makedirs(out_dir, exist_ok=True)
    
    for i, fname in enumerate(to_download):
        print(f"Downloading {i+1}/{len(to_download)}: {fname}")
        kaggle.api.dataset_download_file(dataset, fname, path=out_dir)
        
    # Unzip all downloaded files
    for zf in os.listdir(out_dir):
        if zf.endswith('.zip'):
            zpath = os.path.join(out_dir, zf)
            print(f"Unzipping {zpath}...")
            with zipfile.ZipFile(zpath, 'r') as zip_ref:
                zip_ref.extractall(out_dir)
            os.remove(zpath)

if __name__ == '__main__':
    download_subset()
