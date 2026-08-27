# config.py
"""
Centralna konfiguracja pipeline'u TCGA-BRCA.

DOSTOSUJ TCGA_ROOT DO WLASNEJ SCIEZKI przed pierwszym uruchomieniem.
Pozostale sciezki wyprowadzane sa automatycznie.
"""

import os

# ============================================================================
# SCIEZKI
# ============================================================================

TCGA_ROOT = r"D:\TCGA_Data"

CLINICAL_DIR = os.path.join(TCGA_ROOT, "Clinical")   # plik .txt z GDC
WSI_DIR = os.path.join(TCGA_ROOT, "WSI")             # preparaty .svs
OUTPUT_DIR = os.path.join(TCGA_ROOT, "Pipeline_Output")

PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "processed_images")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
CV_RESULTS_DIR = os.path.join(OUTPUT_DIR, "cv_results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

for _d in (OUTPUT_DIR, PROCESSED_IMAGES_DIR, CACHE_DIR, CV_RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)


# ============================================================================
# PARAMETRY PRZETWARZANIA PREPARATOW
# ============================================================================

class WSIConfig:
    """
    Wartosci odpowiadaja opisowi w rozdziale 2.4 manuskryptu.
    Zmiana ktorejkolwiek uniewaznia zgodnosc z opublikowanymi wynikami.
    """
    PATCH_SIZE = 224            # kafelek 224x224 px, poziom 0 piramidy
    MAX_PATCHES_PER_WSI = 100   # gorny limit na preparat; step2 losuje z tego 40
    THUMBNAIL_SIZE = 512        # miniatura dla trybu 'thumbnail'
    MIN_TISSUE_RATIO = 0.3      # minimalny udzial tkanki w kafelku
    IMAGE_FORMAT = "png"        # png = bezstratny; jpg wprowadzalby artefakty
    NUM_WORKERS = 4             # procesy rownolegle
