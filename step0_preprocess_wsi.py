# 2_preprocess_wsi.py
"""
SKRYPT 2: Preprocessing WSI Images

Zadania:
1. Skanuje folder D:\TCGA_Data\WSI dla .svs files
2. Dla każdego .svs:
   - Extract patches 224x224 LUB
   - Create thumbnail 512x512
3. Skip białe obszary (tissue detection)
4. Zapisuje processed images
5. Tworzy mapping: patient_id → image_paths

Usage:
    python 2_preprocess_wsi.py [--method patches|thumbnail] [--test N]
    
Options:
    --method: 'patches' (default) lub 'thumbnail'
    --test N: Process only first N files (dla testów)
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import config and utilities
try:
    from config import *
    from wsi_utils import process_wsi_file, OPENSLIDE_AVAILABLE
except ImportError:
    print("ERROR: Nie można zaimportować config.py lub utils")
    sys.exit(1)

def scan_wsi_files(wsi_dir: str, extensions: tuple = ('.svs', '.tiff', '.tif')) -> list:
    """
    Skanuje directory dla WSI files
    """
    print(f"\n🔍 Skanowanie: {wsi_dir}")
    
    wsi_dir = Path(wsi_dir)
    
    if not wsi_dir.exists():
        print(f"❌ Folder nie istnieje: {wsi_dir}")
        return []
    
    files = []
    for ext in extensions:
        files.extend(list(wsi_dir.rglob(f'*{ext}')))
    
    files = [str(f) for f in files]
    
    print(f"✅ Znaleziono {len(files)} plików WSI")
    
    # Wyświetl przykładowe nazwy
    if files:
        print("\nPrzykładowe pliki:")
        for f in files[:5]:
            print(f"  - {Path(f).name}")
    
    return files

def process_single_wsi(args_tuple):
    """
    Wrapper function dla multiprocessing
    """
    wsi_path, output_dir, config = args_tuple
    
    try:
        result = process_wsi_file(
            wsi_path=wsi_path,
            output_dir=output_dir,
            method=config['method'],
            patch_size=config['patch_size'],
            max_patches=config['max_patches'],
            thumbnail_size=config['thumbnail_size'],
            min_tissue_ratio=config['min_tissue_ratio'],
            format=config['format']
        )
        return result
    except Exception as e:
        return {
            'wsi_path': wsi_path,
            'success': False,
            'error': str(e)
        }

def process_all_wsi_files(
    wsi_files: list,
    output_dir: str,
    method: str = 'patches',
    num_workers: int = 4,
    test_mode: bool = False,
    test_n: int = 5
) -> pd.DataFrame:
    """
    Przetwarza wszystkie WSI files (z multiprocessing)
    """
    
    if test_mode:
        print(f"\n🧪 TEST MODE: Processing tylko {test_n} plików")
        wsi_files = wsi_files[:test_n]
    
    print(f"\n🔄 Processing {len(wsi_files)} WSI files...")
    print(f"   Method: {method}")
    print(f"   Workers: {num_workers}")
    
    # Prepare config dict for each worker
    config = {
        'method': method,
        'patch_size': WSIConfig.PATCH_SIZE,
        'max_patches': WSIConfig.MAX_PATCHES_PER_WSI,
        'thumbnail_size': WSIConfig.THUMBNAIL_SIZE,
        'min_tissue_ratio': WSIConfig.MIN_TISSUE_RATIO,
        'format': WSIConfig.IMAGE_FORMAT
    }
    
    # Prepare arguments for each file
    args_list = [(wsi_path, output_dir, config) for wsi_path in wsi_files]
    
    results = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_wsi, args): args[0] for args in args_list}
        
        # Progress bar
        pbar = tqdm(total=len(futures), desc="Processing WSI")
        
        for future in as_completed(futures):
            wsi_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error processing {Path(wsi_path).name}: {str(e)}")
                results.append({
                    'wsi_path': wsi_path,
                    'success': False,
                    'error': str(e)
                })
            
            pbar.update(1)
        
        pbar.close()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Statistics
    success_count = df['success'].sum()
    fail_count = len(df) - success_count
    
    print(f"\n📊 Results:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    
    if method == 'patches':
        total_patches = df[df['success']]['num_images'].sum()
        avg_patches = df[df['success']]['num_images'].mean()
        print(f"   📸 Total patches: {int(total_patches)}")
        print(f"   📸 Avg patches per WSI: {avg_patches:.1f}")
    
    return df

def create_image_mapping(results_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Tworzy mapping DataFrame: patient_id → image_paths
    """
    print(f"\n🗂️  Creating image mapping...")
    
    # Filter successful results
    df_success = results_df[results_df['success']].copy()
    
    if df_success.empty:
        print("❌ Brak udanych przetworzonych plików!")
        return pd.DataFrame()
    
    # Expand image_paths (each row may have multiple images)
    records = []
    
    for _, row in df_success.iterrows():
        patient_id = row['patient_id']
        
        if isinstance(row['image_paths'], list):
            image_paths = row['image_paths']
        else:
            # Jeśli string (JSON)
            try:
                image_paths = json.loads(row['image_paths'])
            except:
                image_paths = [row['image_paths']]
        
        for img_path in image_paths:
            records.append({
                'patient_id': patient_id,
                'image_path': img_path,
                'wsi_source': row['wsi_path']
            })
    
    df_mapping = pd.DataFrame(records)
    
    print(f"✅ Created mapping:")
    print(f"   Patients: {df_mapping['patient_id'].nunique()}")
    print(f"   Total images: {len(df_mapping)}")
    print(f"   Avg images per patient: {len(df_mapping) / df_mapping['patient_id'].nunique():.1f}")
    
    # Save mapping
    mapping_path = os.path.join(output_dir, "image_mapping.csv")
    df_mapping.to_csv(mapping_path, index=False)
    print(f"📁 Saved: {mapping_path}")
    
    return df_mapping

def verify_processed_images(df_mapping: pd.DataFrame):
    """
    Weryfikuje czy wszystkie zapisane obrazy istnieją
    """
    print(f"\n🔍 Weryfikacja obrazów...")
    
    missing_count = 0
    
    for img_path in tqdm(df_mapping['image_path'], desc="Checking files"):
        if not os.path.exists(img_path):
            missing_count += 1
    
    if missing_count == 0:
        print(f"✅ Wszystkie {len(df_mapping)} obrazy istnieją!")
    else:
        print(f"⚠️  Brakuje {missing_count} plików")

def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(description='Preprocess TCGA WSI files')
    parser.add_argument('--method', type=str, default='patches', choices=['patches', 'thumbnail'],
                        help='Processing method: patches or thumbnail')
    parser.add_argument('--test', type=int, default=None,
                        help='Test mode: process only N files')
    parser.add_argument('--workers', type=int, default=WSIConfig.NUM_WORKERS,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TCGA WSI PREPROCESSING")
    print("=" * 80)
    
    # Check OpenSlide
    if not OPENSLIDE_AVAILABLE:
        print("\n❌ ERROR: OpenSlide nie jest zainstalowany!")
        print("\nInstalacja:")
        print("  pip install openslide-python")
        print("  Windows: Pobierz binaries z https://openslide.org/download/")
        sys.exit(1)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Method: {args.method}")
    print(f"   WSI Directory: {WSI_DIR}")
    print(f"   Output Directory: {PROCESSED_IMAGES_DIR}")
    
    if args.method == 'patches':
        print(f"   Patch Size: {WSIConfig.PATCH_SIZE}x{WSIConfig.PATCH_SIZE}")
        print(f"   Max Patches per WSI: {WSIConfig.MAX_PATCHES_PER_WSI}")
        print(f"   Min Tissue Ratio: {WSIConfig.MIN_TISSUE_RATIO}")
    else:
        print(f"   Thumbnail Size: {WSIConfig.THUMBNAIL_SIZE}x{WSIConfig.THUMBNAIL_SIZE}")
    
    # Step 1: Scan WSI files
    wsi_files = scan_wsi_files(WSI_DIR)
    
    if not wsi_files:
        print("\n❌ Nie znaleziono plików WSI!")
        print(f"Sprawdź ścieżkę: {WSI_DIR}")
        return
    
    # Step 2: Process all WSI files
    test_mode = args.test is not None
    test_n = args.test if test_mode else 0
    
    results_df = process_all_wsi_files(
        wsi_files=wsi_files,
        output_dir=PROCESSED_IMAGES_DIR,
        method=args.method,
        num_workers=args.workers,
        test_mode=test_mode,
        test_n=test_n
    )
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "wsi_processing_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📁 Results saved: {results_path}")
    
    # Step 3: Create image mapping
    df_mapping = create_image_mapping(results_df, OUTPUT_DIR)
    
    if df_mapping.empty:
        print("\n❌ Nie udało się stworzyć image mapping!")
        return
    
    # Step 4: Verify images
    verify_processed_images(df_mapping)
    
    print(f"\n✅ SUKCES!")
    print(f"📁 Processed images: {PROCESSED_IMAGES_DIR}")
    print(f"📁 Image mapping: {os.path.join(OUTPUT_DIR, 'image_mapping.csv')}")
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"   Total WSI processed: {results_df['success'].sum()}")
    print(f"   Unique patients: {df_mapping['patient_id'].nunique()}")
    print(f"   Total images: {len(df_mapping)}")
    
    print("\n" + "=" * 80)
    print("NASTĘPNY KROK: python 3_match_and_split_data.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
