# utils/wsi_utils.py
"""
Utilities do przetwarzania WSI (Whole Slide Images) w formacie .svs
Używa OpenSlide do wczytywania i extractowania patches/thumbnails
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import cv2

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("WARNING: OpenSlide nie jest zainstalowany!")
    print("Instalacja: pip install openslide-python")
    print("Windows: Pobierz binaries z https://openslide.org/download/")

def check_openslide():
    """Sprawdź czy OpenSlide jest dostępny"""
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("OpenSlide nie jest zainstalowany. Zainstaluj openslide-python.")

def get_tissue_mask(thumbnail: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Tworzy maskę tissue (oddziela tissue od białego tła)
    
    Args:
        thumbnail: RGB thumbnail image
        threshold: piksele jaśniejsze niż to = background
    
    Returns:
        Binary mask (True = tissue, False = background)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    
    # Threshold: białe obszary to background
    tissue_mask = gray < threshold
    
    # Morphological operations dla oczyszczenia
    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    
    return tissue_mask.astype(bool)

def calculate_tissue_ratio(patch: np.ndarray, threshold: int = 220) -> float:
    """
    Oblicza stosunek tissue do całej powierzchni patch
    
    Args:
        patch: RGB patch
        threshold: piksele jaśniejsze niż to = background
    
    Returns:
        Ratio (0.0 - 1.0) ile tissue jest w patch
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    tissue_pixels = np.sum(gray < threshold)
    total_pixels = gray.size
    return tissue_pixels / total_pixels

class WSIReader:
    """Klasa do wczytywania i przetwarzania WSI files"""
    
    def __init__(self, wsi_path: str):
        """
        Args:
            wsi_path: Ścieżka do .svs file
        """
        check_openslide()
        self.wsi_path = Path(wsi_path)
        self.slide = openslide.OpenSlide(str(wsi_path))
        
        # Properties
        self.dimensions = self.slide.dimensions  # (width, height) at level 0
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        
        print(f"Loaded WSI: {self.wsi_path.name}")
        print(f"  Dimensions: {self.dimensions[0]}x{self.dimensions[1]}")
        print(f"  Levels: {self.level_count}")
    
    def get_thumbnail(self, size: int = 512) -> np.ndarray:
        """
        Zwraca thumbnail WSI
        
        Args:
            size: max dimension of thumbnail
        
        Returns:
            RGB thumbnail as numpy array
        """
        thumb = self.slide.get_thumbnail((size, size))
        return np.array(thumb)
    
    def extract_patch(self, x: int, y: int, size: int, level: int = 0) -> np.ndarray:
        """
        Ekstraktuje pojedynczy patch z WSI
        
        Args:
            x, y: Top-left coordinates (at level 0)
            size: Patch size (square)
            level: Pyramid level to extract from
        
        Returns:
            RGB patch as numpy array
        """
        # Read region zwraca RGBA
        patch = self.slide.read_region((x, y), level, (size, size))
        # Convert RGBA -> RGB
        patch = patch.convert('RGB')
        return np.array(patch)
    
    def extract_patches_grid(
        self, 
        patch_size: int = 224, 
        overlap: int = 0,
        level: int = 0,
        min_tissue_ratio: float = 0.3,
        max_patches: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Ekstraktuje patches w grid pattern
        
        Args:
            patch_size: Size of each patch
            overlap: Overlap between patches (in pixels)
            level: Pyramid level
            min_tissue_ratio: Minimum tissue ratio to keep patch
            max_patches: Maximum number of patches to extract
        
        Returns:
            List of (patch_array, x_coord, y_coord)
        """
        width, height = self.level_dimensions[level]
        step = patch_size - overlap
        
        # Get thumbnail for tissue detection
        thumbnail = self.get_thumbnail(1024)
        tissue_mask = get_tissue_mask(thumbnail)
        
        # Calculate thumbnail to level 0 scale
        thumb_scale_x = self.dimensions[0] / thumbnail.shape[1]
        thumb_scale_y = self.dimensions[1] / thumbnail.shape[0]
        
        patches = []
        
        for y in range(0, height - patch_size, step):
            for x in range(0, width - patch_size, step):
                # Check if this region has tissue (using thumbnail)
                thumb_x = int(x / thumb_scale_x)
                thumb_y = int(y / thumb_scale_y)
                thumb_size = max(1, int(patch_size / thumb_scale_x))
                
                # Extract corresponding region from tissue mask
                mask_region = tissue_mask[
                    thumb_y:thumb_y+thumb_size, 
                    thumb_x:thumb_x+thumb_size
                ]
                
                if mask_region.size == 0:
                    continue
                
                tissue_ratio = np.mean(mask_region)
                
                if tissue_ratio < min_tissue_ratio:
                    continue  # Skip patches with mostly background
                
                # Extract patch
                patch = self.extract_patch(x, y, patch_size, level)
                
                # Double-check tissue ratio on actual patch
                actual_ratio = calculate_tissue_ratio(patch)
                if actual_ratio < min_tissue_ratio:
                    continue
                
                patches.append((patch, x, y))
                
                if max_patches and len(patches) >= max_patches:
                    return patches
        
        return patches
    
    def extract_patches_random(
        self,
        patch_size: int = 224,
        num_patches: int = 100,
        level: int = 0,
        min_tissue_ratio: float = 0.3,
        max_attempts: int = 1000
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Ekstraktuje patches w random locations (zamiast grid)
        Szybsze dla dużych WSI
        
        Args:
            patch_size: Size of each patch
            num_patches: Number of patches to extract
            level: Pyramid level
            min_tissue_ratio: Minimum tissue ratio
            max_attempts: Max attempts to find valid patches
        
        Returns:
            List of (patch_array, x_coord, y_coord)
        """
        width, height = self.level_dimensions[level]
        
        # Get thumbnail for tissue detection
        thumbnail = self.get_thumbnail(1024)
        tissue_mask = get_tissue_mask(thumbnail)
        
        thumb_scale_x = self.dimensions[0] / thumbnail.shape[1]
        thumb_scale_y = self.dimensions[1] / thumbnail.shape[0]
        
        patches = []
        attempts = 0
        
        while len(patches) < num_patches and attempts < max_attempts:
            # Random location
            x = np.random.randint(0, width - patch_size)
            y = np.random.randint(0, height - patch_size)
            
            # Check tissue ratio in thumbnail
            thumb_x = int(x / thumb_scale_x)
            thumb_y = int(y / thumb_scale_y)
            thumb_size = max(1, int(patch_size / thumb_scale_x))
            
            mask_region = tissue_mask[
                thumb_y:min(thumb_y+thumb_size, tissue_mask.shape[0]), 
                thumb_x:min(thumb_x+thumb_size, tissue_mask.shape[1])
            ]
            
            if mask_region.size == 0:
                attempts += 1
                continue
            
            tissue_ratio = np.mean(mask_region)
            
            if tissue_ratio < min_tissue_ratio:
                attempts += 1
                continue
            
            # Extract patch
            patch = self.extract_patch(x, y, patch_size, level)
            
            # Double-check
            actual_ratio = calculate_tissue_ratio(patch)
            if actual_ratio < min_tissue_ratio:
                attempts += 1
                continue
            
            patches.append((patch, x, y))
            attempts += 1
        
        print(f"Extracted {len(patches)} patches in {attempts} attempts")
        return patches
    
    def close(self):
        """Zamknij slide"""
        self.slide.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def save_patches(
    patches: List[Tuple[np.ndarray, int, int]], 
    output_dir: Path, 
    patient_id: str,
    format: str = 'png'
) -> List[str]:
    """
    Zapisuje patches do plików
    
    Args:
        patches: Lista (patch, x, y)
        output_dir: Output directory
        patient_id: Patient ID dla naming
        format: 'png' lub 'jpg'
    
    Returns:
        Lista ścieżek do zapisanych plików
    """
    output_dir = Path(output_dir)
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for idx, (patch, x, y) in enumerate(patches):
        filename = f"{patient_id}_patch_{idx:04d}_x{x}_y{y}.{format}"
        filepath = patient_dir / filename
        
        img = Image.fromarray(patch)
        if format == 'jpg':
            img.save(filepath, quality=95)
        else:
            img.save(filepath)
        
        saved_paths.append(str(filepath))
    
    return saved_paths

def process_wsi_file(
    wsi_path: str,
    output_dir: str,
    method: str = 'patches',
    patch_size: int = 224,
    max_patches: int = 100,
    thumbnail_size: int = 512,
    min_tissue_ratio: float = 0.3,
    format: str = 'png'
) -> Dict:
    """
    Główna funkcja do przetwarzania pojedynczego WSI file
    
    Args:
        wsi_path: Path do .svs file
        output_dir: Output directory
        method: 'patches' lub 'thumbnail'
        patch_size: Size for patches
        max_patches: Max patches to extract
        thumbnail_size: Size for thumbnail
        min_tissue_ratio: Min tissue ratio
        format: Output format
    
    Returns:
        Dict z informacjami o przetworzonym pliku
    """
    wsi_path = Path(wsi_path)
    patient_id = wsi_path.stem.split('.')[0]  # Extract patient ID from filename
    
    # Extract TCGA ID if present
    import re
    match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', patient_id, re.IGNORECASE)
    if match:
        patient_id = match.group(0).upper()
    
    result = {
        'patient_id': patient_id,
        'wsi_path': str(wsi_path),
        'method': method,
        'image_paths': [],
        'success': False
    }
    
    try:
        with WSIReader(str(wsi_path)) as wsi:
            if method == 'thumbnail':
                # Create thumbnail
                thumbnail = wsi.get_thumbnail(thumbnail_size)
                
                # Save thumbnail
                output_dir = Path(output_dir)
                patient_dir = output_dir / patient_id
                patient_dir.mkdir(parents=True, exist_ok=True)
                
                thumb_path = patient_dir / f"{patient_id}_thumbnail.{format}"
                Image.fromarray(thumbnail).save(thumb_path)
                
                result['image_paths'] = [str(thumb_path)]
                
            elif method == 'patches':
                # Extract patches
                patches = wsi.extract_patches_random(
                    patch_size=patch_size,
                    num_patches=max_patches,
                    min_tissue_ratio=min_tissue_ratio
                )
                
                # Save patches
                saved_paths = save_patches(patches, output_dir, patient_id, format)
                result['image_paths'] = saved_paths
            
            result['success'] = True
            result['num_images'] = len(result['image_paths'])
    
    except Exception as e:
        print(f"Error processing {wsi_path}: {str(e)}")
        result['error'] = str(e)
    
    return result

if __name__ == "__main__":
    # Test WSI processing
    import sys
    
    if len(sys.argv) > 1:
        wsi_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"
        
        print(f"Processing: {wsi_path}")
        result = process_wsi_file(wsi_path, output_dir, method='patches', max_patches=10)
        
        print("\n=== Result ===")
        print(f"Success: {result['success']}")
        print(f"Patient ID: {result['patient_id']}")
        print(f"Images extracted: {result['num_images']}")
    else:
        print("Usage: python wsi_utils.py <wsi_path> [output_dir]")
