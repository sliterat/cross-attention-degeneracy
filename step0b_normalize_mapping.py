#!/usr/bin/env python3
"""
step0b_normalize_mapping.py — brakujace ogniwo miedzy przetwarzaniem preparatow
a wlasciwym pipeline'em.

PROBLEM

step0_preprocess_wsi.py zapisuje `image_mapping.csv` o kolumnach:
    patient_id, image_path, wsi_source

Natomiast step1b, step1 i step2 oczekuja `wsi_processed_mapping.csv` o kolumnach:
    patient_id, image_path, wsi_source, patient_id_norm, processed_path, original_path

Bez tego kroku lancuch jest przerwany i pipeline nie da sie uruchomic od zera.
Skrypt dopisuje trzy brakujace kolumny:

    patient_id_norm  identyfikator pacjenta w postaci kanonicznej (wielkie litery)
    processed_path   sciezka do kafelka (kopia image_path)
    original_path    sciezka do preparatu zrodlowego (kopia wsi_source)

Kolumna `original_path` jest wykorzystywana przez step1b do odczytu barcode'u
TCGA, wiec musi wskazywac plik .svs, a nie kafelek.

Uzycie:
    python step0b_normalize_mapping.py
    python step0b_normalize_mapping.py --inp image_mapping.csv --out wsi_processed_mapping.csv
"""

import os
import re
import argparse
import pandas as pd

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\TCGA_Data\Pipeline_Output"

TCGA_ID = re.compile(r"TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(OUTPUT_DIR, "image_mapping.csv"))
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "wsi_processed_mapping.csv"))
    args = ap.parse_args()

    print("=" * 74)
    print("NORMALIZACJA TABELI MAPOWANIA")
    print("=" * 74)

    if not os.path.exists(args.inp):
        raise SystemExit(f"Brak {args.inp}\nUruchom najpierw step0_preprocess_wsi.py")

    m = pd.read_csv(args.inp)
    print(f"\n  Wczytano: {len(m)} kafelkow, kolumny: {list(m.columns)}")

    for col in ("patient_id", "image_path", "wsi_source"):
        if col not in m.columns:
            raise SystemExit(f"Plik wejsciowy nie zawiera kolumny '{col}'.")

    m["patient_id_norm"] = m["patient_id"].astype(str).str.upper().str.strip()
    m["processed_path"] = m["image_path"]
    m["original_path"] = m["wsi_source"]

    # Kontrola: identyfikator musi dac sie odczytac z nazwy preparatu zrodlowego,
    # inaczej step1b nie rozpozna typu materialu.
    src = m["original_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
    found = src.str.extract(f"({TCGA_ID.pattern})")[0].str.upper()
    bad = found.isna().sum()
    mism = (found.dropna() != m.loc[found.notna(), "patient_id_norm"]).sum()

    print(f"\n--- Kontrola spojnosci ---")
    print(f"  Kafelkow bez rozpoznanego barcode'u : {bad}")
    print(f"  Niezgodnosci pacjent <-> preparat   : {mism}")
    if bad:
        print("  !! step1b odrzuci te kafelki. Sprawdz nazwy plikow .svs.")
    if mism:
        print("  !! BLAD KRYTYCZNY: kafelki przypisane do niewlasciwych pacjentow.")
        raise SystemExit(1)
    if not bad and not mism:
        print("  OK")

    print(f"\n  Pacjentow: {m.patient_id_norm.nunique()}")
    print(f"  Kafelkow na pacjenta: mediana "
          f"{m.groupby('patient_id_norm').size().median():.0f}")

    m.to_csv(args.out, index=False)
    print("\n" + "=" * 74)
    print("ZAPISANO")
    print("=" * 74)
    print(f"  {args.out}")
    print("\n  NASTEPNY KROK:  python step1b_filter_slides.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
