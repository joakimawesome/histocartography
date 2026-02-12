import os
import argparse
import json
import pandas as pd
from pathlib import Path
import logging

def aggregate_qc_metrics(output_dir: str, result_file: str):
    """
    Crawls output_dir for */qc/qc_metrics.json and aggregates them.
    """
    root = Path(output_dir)
    qc_files = list(root.glob("*/qc/qc_metrics.json"))
    
    if not qc_files:
        print(f"No qc_metrics.json files found in {output_dir}")
        return

    records = []
    
    for qf in qc_files:
        slide_id = qf.parent.parent.name
        try:
            with open(qf, 'r') as f:
                data = json.load(f)
            data['slide_id'] = slide_id
            records.append(data)
        except Exception as e:
            print(f"Error reading {qf}: {e}")
            
    if records:
        df = pd.DataFrame(records)
        # Reorder columns to put slide_id first
        cols = ['slide_id'] + [c for c in df.columns if c != 'slide_id']
        df = df[cols]
        
        df.to_csv(result_file, index=False)
        print(f"Aggregated QC metrics for {len(records)} slides to {result_file}")
    else:
        print("No valid QC records found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate QC metrics from pipeline outputs.")
    parser.add_argument("--output_dir", required=True, help="Root directory containing slide subdirectories.")
    parser.add_argument("--result_file", default="qc_summary.csv", help="Path to save the aggregated CSV.")
    
    args = parser.parse_args()
    
    aggregate_qc_metrics(args.output_dir, args.result_file)
