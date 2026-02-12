import torch
import os
import shutil
from torch_geometric.data import Data
from histocartography_ext.features.handcrafted import extract_graph_stats
from histocartography_ext.features.gnn import extract_gnn_embeddings, GraphEncoder

def create_dummy_graph():
    num_nodes = 20
    num_edges = 50
    feature_dim = 10
    
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.rand(num_edges, 1) # Distances
    pos = torch.rand(num_nodes, 2)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.mpp = 0.5
    data.coord_space = 'level-0'
    return data

def verify_handcrafted(data):
    print("\n--- Verifying Handcrafted Features ---")
    stats = extract_graph_stats(data)
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # Simple assertions
    assert 'num_nodes' in stats
    assert stats['num_nodes'] == 20
    assert 'degree_mean' in stats
    print("Handcrafted verification passed!")

def verify_gnn(data):
    print("\n--- Verifying GNN Embeddings ---")
    # 1. Random model
    emb = extract_gnn_embeddings(data)
    print(f"Random model embedding shape: {emb.shape}")
    assert emb.shape == (128,) or emb.shape == (1, 128)
    
    # 2. Custom model
    model = GraphEncoder(input_dim=10, hidden_dim=32, output_dim=64, num_layers=2)
    emb_custom = extract_gnn_embeddings(data, model=model)
    print(f"Custom model embedding shape: {emb_custom.shape}")
    assert emb_custom.shape == (64,) or emb_custom.shape == (1, 64)
    print("GNN verification passed!")

def verify_cli():
    print("\n--- Verifying CLI ---")
    # Setup dummy directory
    test_dir = 'test_graphs_temp'
    os.makedirs(test_dir, exist_ok=True)
    
    data = create_dummy_graph()
    torch.save(data, os.path.join(test_dir, 'slide_1.pt'))
    torch.save(data, os.path.join(test_dir, 'slide_2.pt'))
    
    out_file = 'test_features.parquet'
    
    # Run CLI command (simulated)
    cmd = f"python -m histocartography_ext.features.extract --graph_dir {test_dir} --out {out_file} --mode stats"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    if os.path.exists(out_file):
        print("Success: Output file created.")
        # Optional: load and check
        import pandas as pd
        if out_file.endswith('.parquet'):
             df = pd.read_parquet(out_file)
        else:
             df = pd.read_csv(out_file)
        print("Loaded DataFrame:")
        print(df.head())
        assert len(df) == 2
        os.remove(out_file)
    else:
        print("Error: Output file not created.")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    data = create_dummy_graph()
    verify_handcrafted(data)
    verify_gnn(data)
    # verify_cli() # Run in separate step if needed, or here
    verify_cli()
