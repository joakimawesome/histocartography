import argparse
import os
import glob
import torch
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from .handcrafted import extract_graph_stats
from .gnn import extract_gnn_embeddings, GraphEncoder

def load_graph(path):
    # Try loading with torch.load
    try:
        data = torch.load(path, weights_only=False)
        return data
    except TypeError:
        # Fallback for older torch versions
        data = torch.load(path)
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract features from WSI graphs.")
    parser.add_argument("--graph_dir", type=str, required=True, help="Directory containing .pt/.pth graph files.")
    parser.add_argument("--out", type=str, required=True, help="Output file path (.parquet or .npy).")
    parser.add_argument("--mode", type=str, choices=['stats', 'gnn'], required=True, help="Feature extraction mode.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to GNN model checkpoint (for GNN mode).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.graph_dir):
        raise ValueError(f"Graph directory does not exist: {args.graph_dir}")
        
    # Find all graph files
    graph_files = glob.glob(os.path.join(args.graph_dir, "*.pt")) + \
                  glob.glob(os.path.join(args.graph_dir, "*.pth"))
    
    if not graph_files:
        print(f"No graph files found in {args.graph_dir}")
        # Create empty file or just exit?
        return

    results = []
    
    # Load model if GNN mode
    model = None
    if args.mode == 'gnn':
        if args.model_path:
            # Here we assume a specific model architecture or save/load method.
            # For simplicity in this CLI, we might just load state dict if compatible,
            # or assume the whole model was saved.
            # Let's try loading the whole object first, or fallback.
            try:
                loaded = torch.load(args.model_path)
                if isinstance(loaded, dict):
                     # It's a state dict, we need to know arch.
                     # For now, let's warn and use random or fail.
                     print("Warning: Model path seems to be a state_dict. "
                           "Please ensure 'GraphEncoder' is instantiated correctly. "
                           "Using random initialization for now if architecture not known.")
                     model = None # Fallback to random in extract_gnn_embeddings for now
                elif isinstance(loaded, torch.nn.Module):
                    model = loaded
                    model.eval()
                else:
                    print(f"Unknown model object type: {type(loaded)}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        else:
            # Random initialized model will be created per graph unless we create one here.
            # Ideally create one consistent random model here.
            pass

    for path in tqdm(graph_files, desc="Extracting features"):
        data = load_graph(path)
        if data is None:
            continue
            
        slide_id = os.path.splitext(os.path.basename(path))[0]
        
        row = {'slide_id': slide_id}
        
        # Extract Metadata
        if hasattr(data, 'mpp'):
            row['mpp'] = data.mpp
        if hasattr(data, 'coord_space'):
            row['coord_space'] = data.coord_space
            
        if args.mode == 'stats':
            stats = extract_graph_stats(data)
            row.update(stats)
            
        elif args.mode == 'gnn':
            # Initialize random model once if not loaded
            if model is None:
                 input_dim = data.x.shape[1] if data.x is not None else 1
                 # Use default arch
                 model = GraphEncoder(input_dim=input_dim, hidden_dim=64, output_dim=128)
                 model.eval()
            
            emb = extract_gnn_embeddings(data, model=model)
            # Flatten if needed, should be (1, D) or (D,)
            if emb.ndim > 1:
                emb = emb.flatten()
            
            for i, val in enumerate(emb):
                row[f'feat_{i}'] = val
        
        results.append(row)
    
    if not results:
        print("No results extracted.")
        return

    df = pd.DataFrame(results)
    
    # Save output
    if args.out.endswith('.parquet'):
        try:
             df.to_parquet(args.out)
        except ImportError:
             print("pyarrow or fastparquet required for parquet export. Falling back to CSV.")
             df.to_csv(args.out.replace('.parquet', '.csv'), index=False)
    elif args.out.endswith('.npy'):
        # Save structured array
        np.save(args.out, df.to_records(index=False))
    else:
        # Default to csv
        df.to_csv(args.out, index=False)
        
    print(f"Saved features to {args.out}")

if __name__ == "__main__":
    main()
