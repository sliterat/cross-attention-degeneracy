# Bidirectional cross-attention in histopathology–clinical data fusion

Source code for the manuscript *"Bidirectional cross-attention does not perform* *cross-modal interaction: architectural degeneracy and variance decomposition in* *histopathology–clinical data fusion."*

Author: Tomasz Pruś (independent researcher) Code licence: MIT · Manuscript and figures: CC BY 4.0


## What this work shows

Three findings, ordered by strength of evidence:

1. **Architectural degeneracy.** The cross-attention direction in which the image acts as query and the clinical vector as key and value is mathematically degenerate: with a single-element key the softmax evaluates identically to 1, so the module output is **bitwise identical for arbitrarily different images**. Proven analytically, confirmed numerically.

2. **Late fusion outperforms early fusion.** Rank averaging of two unimodal model predictions (AUC 0.772) is the only variant that beats all others after Holm correction.

3. **Variance decomposition.** Across 450 trained models: data splitting accounts for 86.2% of AUC variability, weight initialisation for 11.5%, architecture for 2.3%. Initialisation variance exceeds architectural variance 5.1-fold.


## Requirements

```
pip install -r requirements.txt
```

**Windows note:** `openslide-python` additionally requires the OpenSlide binaries. Download from [https://openslide.org/download/](https://openslide.org/download/), extract, and add the `bin` directory to `PATH`. Without this, step 0 will not run (all other steps will).

All computation was performed on CPU. Caching the image representations (step 2) keeps steps 3–4 in the range of minutes rather than hours, even without an accelerator.


## Data

Not included — download from the Genomic Data Commons:

- **Clinical data:** `nationwidechildrens.org\_clinical\_patient\_brca.txt` → place in `CLINICAL\_DIR`

- **Slides:** TCGA-BRCA `.svs` files → place in `WSI\_DIR`

All paths are set in one place: the `TCGA\_ROOT` variable in `config.py`.


## Pipeline order

Each step prints the name of the next one on completion.

```
\# STEP 0 — slide tiling (requires OpenSlide; several hours)  
python step0\_preprocess\_wsi.py --method patches  
  
\# STEP 0b — reconcile the mapping table schema  
python step0b\_normalize\_mapping.py  
  
\# STEP 1b — material filtering by TCGA barcode (excludes normal tissue)  
python step1b\_filter\_slides.py  
  
\# STEP 1 — clinical dataset construction  
python step1\_build\_dataset.py --mapping "\<OUTPUT\_DIR\>\\wsi\_processed\_mapping\_FILTERED.csv"  
  
\# STEP 2 — image representation precomputation (frozen EfficientNetB0; several hours)  
python step2\_precompute\_embeddings.py --mapping "\<OUTPUT\_DIR\>\\wsi\_processed\_mapping\_FILTERED.csv"  
  
\# STEP 3 — training and evaluation: 450 models (~1 h)  
python step3\_train\_eval\_cv.py --target er --deterministic --seeds 3 ^  
       --arms tab img concat concat\_wide cross cross\_a2b --tag final --save-preds  
  
\# STEP 4 — late fusion and modality agreement  
python step4\_late\_fusion.py --preds "\<OUTPUT\_DIR\>\\cv\_results\\cv\_preds\_er\_final.csv" --out figures  
  
\# FIGURES AND TABLES  
python make\_fig1\_fig2.py --out figures --dpi 600 --lang en  
python make\_figures.py --results cv\_results --out figures --dpi 600 --lang en
```

Steps 0 and 2 support `--resume`. Figure scripts accept `--lang pl` for Polish captions (English is selected with `--lang en`).


## Pipeline at a glance

| Step | Script | Output |
| - | - | - |
| 0 | `step0\_preprocess\_wsi.py` | PNG patches + `image\_mapping.csv` |
| 0b | `step0b\_normalize\_mapping.py` | `wsi\_processed\_mapping.csv` |
| 1b | `step1b\_filter\_slides.py` | `wsi\_processed\_mapping\_FILTERED.csv` |
| 1 | `step1\_build\_dataset.py` | `dataset\_FIXED.csv`, `feature\_columns.csv` |
| 2 | `step2\_precompute\_embeddings.py` | `cache/embeddings.npy` (~1.3 GB) |
| 3 | `step3\_train\_eval\_cv.py` | `cv\_folds\_\*.csv`, `cv\_preds\_\*.csv` |
| 4 | `step4\_late\_fusion.py` | Figures 7–9, Tables S4–S6 |
| — | `make\_fig1\_fig2.py`, `make\_figures.py` | Figures 1–6, Tables S1–S3 |



## Reproducibility

The `--deterministic` flag in step 3 enables `tf.config.experimental.enable\_op\_determinism()`, sets `TF\_DETERMINISTIC\_OPS` and `TF\_CUDNN\_DETERMINISTIC` **before importing TensorFlow**, and restricts execution to a single thread. Verified: two independent runs of the full experiment produced bitwise identical results across all 450 models.

Without the flag, repeated runs differ by up to 0.20 AUC at the level of individual folds — which is itself one of the findings reported in the paper (Section 3.6).

**Reproducibility was verified for the package versions pinned in `requirements.txt`.** Other TensorFlow versions may produce different values.

**cv\_preds\_er\_final.csv** — patient-level predictions for all 450 models. This file allows you to reproduce the late fusion analysis (step4\_late\_fusion.py) without having to rerun step 3.



## Methodological safeguards built into the code

The code contains checks that arose from errors found during development. Worth knowing before modifying anything:

- **Label leakage test** (step 1): a decision tree trained on features alone must score *below* the majority-class baseline. A result above 0.95 indicates leakage and halts the pipeline.

- **Patient ↔ slide assignment check** (step 1b): the identifier parsed from each `.svs` filename is compared against the mapping table; any mismatch aborts execution.

- **Normal-tissue exclusion** (step 1b): without this filter the model learns to predict tumour receptor status from images of healthy tissue (12.5% of patches in the raw TCGA-BRCA cohort).

- **ASCO/CAP HER2 adjudication** (step 1): FISH takes precedence over IHC, and equivocal cases are excluded rather than assigned arbitrarily.

- **Patient-level evaluation** (step 3): aggregation precedes metric computation. Per-patch evaluation inflates results and understates variance.

- **Result file protection** (step 3): a file with a mismatched column schema is archived rather than overwritten. This does not, however, protect against concatenating two runs with the **same** schema — use distinct `--tag` values.


## Output directory structure

```
Pipeline\_Output/  
├── processed\_images/            PNG patches (step 0)  
├── cache/  
│   ├── embeddings.npy           representation cache, float16  
│   └── embeddings\_index.csv  
├── cv\_results/  
│   ├── cv\_folds\_er\_final.csv    metrics, 450 rows  
│   └── cv\_preds\_er\_final.csv    patient-level predictions  
├── figures/                     Figures 1–9, Tables S1–S6  
├── wsi\_processed\_mapping.csv  
├── wsi\_processed\_mapping\_FILTERED.csv  
├── dataset\_FIXED.csv  
├── feature\_columns.csv  
└── table\_material\_composition.csv
```


## Citation

```
Pruś T. Bidirectional cross-attention does not perform cross-modal interaction:  
architectural degeneracy and variance decomposition in histopathology–clinical  
data fusion. \[year\]. DOI: \[to be completed\]
```

## Contact

[prus.tomasz.1972@gmail.com](mailto:prus.tomasz.1972@gmail.com)

