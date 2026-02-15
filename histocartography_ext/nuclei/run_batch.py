import argparse
import os
import glob
import traceback
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use relative import if running as module, otherwise absolute
try:
    from .segmentation import segment_nuclei
except ImportError:
    from histocartography_ext.nuclei.segmentation import segment_nuclei

def process_slide(
    slide_path, 
    out_dir, 
    model_path, 
    level, 
    tile_size, 
    overlap, 
    device,
    batch_size
):
    try:
        slide_name = Path(slide_path).stem
        out_path = os.path.join(out_dir, f"{slide_name}.parquet")
        
        if os.path.exists(out_path):
            return f"Skipped {slide_name} (exists)"

        # Run segmentation
        _instance_map, df = segment_nuclei(
            slide_path=slide_path,
            level=level,
            tile_size=tile_size,
            overlap=overlap,
            model_path=model_path,
            device=device,
            batch_size=batch_size
        )
        
        # Save
        if not df.empty:
            df.to_parquet(out_path, index=False)
        else:
            # Create empty parquet with schema?
            pass
            
        return f"Processed {slide_name}: {len(df)} nuclei"
        
    except Exception as e:
        return f"Error processing {slide_path}: {str(e)}\n{traceback.format_exc()}"

def main():
    parser = argparse.ArgumentParser(description="Batch Nucleus Segmentation")
    parser.add_argument("--wsi_dir", required=True, help="Directory containing WSIs")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--model_path", required=True, help="Path to HoVer-Net checkpoint")
    parser.add_argument("--level", type=int, default=0, help="WSI level")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (processes)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per worker")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--extensions", nargs="+", default=["*.svs", "*.tif", "*.ndpi", "*.mrxs"], help="File extensions")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Collect files
    files = []
    for ext in args.extensions:
        files.extend(glob.glob(os.path.join(args.wsi_dir, ext)))
        # Also try recursive?
        files.extend(glob.glob(os.path.join(args.wsi_dir, "**", ext), recursive=True))
        
    files = sorted(list(set(files)))
    print(f"Found {len(files)} slides.")
    
    # Multiprocessing vs Sequential
    # Since we use GPU, multiprocessing might be tricky unless we use 'spawn' and manage GPU memory.
    # Safe default: Sequential if workers=1, or use ProcessPoolExecutor if workers > 1.
    
    if args.workers == 1:
        for f in tqdm(files):
            res = process_slide(
                f, args.out_dir, args.model_path, args.level, 
                args.tile_size, args.overlap, args.device, args.batch_size
            )
            print(res)
    else:
        # Note: CUDA with multiprocessing requires 'spawn'
        if args.device == 'cuda':
             import multiprocessing
             multiprocessing.set_start_method('spawn', force=True)
             
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_slide, 
                    f, args.out_dir, args.model_path, args.level, 
                    args.tile_size, args.overlap, args.device, args.batch_size
                ): f for f in files
            }
            
            for future in tqdm(as_completed(futures), total=len(files)):
                print(future.result())

if __name__ == "__main__":
    main()
