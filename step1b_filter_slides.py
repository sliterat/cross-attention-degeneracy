#!/usr/bin/env python3
"""
step1b_filter_slides.py — filtrowanie preparatów wg barcode'u TCGA

DLACZEGO TEN KROK JEST KONIECZNY

Barcode TCGA koduje pochodzenie materiału na pozycjach po identyfikatorze pacjenta:

    TCGA-AC-A2FG-01A-01-TSA.BE090090-....svs
              ^^^^ ^^ ^^^
              |    |  └─ typ preparatu
              |    └──── porcja
              └───────── typ materiału

Typ materiału (Sample Type Code):
    01  guz pierwotny
    06  przerzut
    11  TKANKA PRAWIDŁOWA

Typ preparatu:
    DX          diagnostyczny, utrwalony w formalinie i zatopiony w parafinie (FFPE)
    TS, BS, MS  skrawek mrożony (top / bottom / middle)

Bez filtrowania model uczy się przewidywać status receptorów guza z obrazów
tkanki prawidłowej, w której guza fizycznie nie ma. W kohorcie TCGA-BRCA
dotyczy to około 12% kafelków i kilkudziesięciu pacjentek, dla których
dostępne są WYŁĄCZNIE preparaty tkanki prawidłowej.

Skrawki mrożone są materiałem gorszej jakości (artefakty kryształów lodu,
odmienna charakterystyka barwienia), ale guz w nich występuje. Domyślnie
są zachowywane; flaga --only-dx pozwala je odrzucić kosztem znacznej
redukcji kohorty.

Wyjście: przefiltrowany plik mapowania oraz tabela składu materiału
do zamieszczenia w publikacji.

Kolejność uruchamiania:
    python step1b_filter_slides.py          <- TEN KROK
    python step1_build_dataset.py --mapping <przefiltrowany plik>
    python step2_precompute_embeddings.py --mapping <przefiltrowany plik>
    python step3_train_eval_cv.py ...

Użycie:
    python step1b_filter_slides.py
    python step1b_filter_slides.py --only-dx          # tylko FFPE diagnostyczne
    python step1b_filter_slides.py --keep-metastatic  # zachowaj przerzuty
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\TCGA_Data\Pipeline_Output"

BARCODE = re.compile(
    r"(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})"     # pacjent
    r"-(\d{2})([A-Za-z])"                        # typ materiału + fiolka
    r"-(\d{2})"                                  # porcja
    r"-([A-Za-z]{2})(\d*)"                       # typ preparatu + numer
)

SAMPLE_NAME = {
    "01": "guz pierwotny", "02": "guz nawrotowy", "05": "nowotwór pierwotny dodatkowy",
    "06": "przerzut", "07": "przerzut dodatkowy", "10": "krew prawidłowa",
    "11": "tkanka prawidłowa", "12": "tkanka policzka", "14": "szpik prawidłowy",
}
SLIDE_NAME = {
    "DX": "FFPE diagnostyczny", "TS": "mrożony (górny)",
    "BS": "mrożony (dolny)", "MS": "mrożony (środkowy)",
}
FROZEN = {"TS", "BS", "MS"}


def parse_barcode(path):
    """Wyciąga składowe barcode'u z nazwy pliku preparatu."""
    if not isinstance(path, str):
        return pd.Series([None] * 5)
    name = path.replace("\\", "/").split("/")[-1]
    m = BARCODE.match(name)
    if not m:
        return pd.Series([None] * 5)
    pid, sample, vial, portion, slide = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    return pd.Series([pid.upper(), sample, vial, portion, slide.upper()])


def composition_table(df, label):
    """Tabela składu materiału — do zamieszczenia w publikacji."""
    t = (df.groupby(["slide_type", "sample_type"])
           .agg(kafelki=("processed_path", "size"),
                pacjentki=("patient_id_norm", "nunique"))
           .reset_index())
    t["typ_preparatu"] = t.slide_type.map(lambda s: SLIDE_NAME.get(s, s))
    t["typ_materialu"] = t.sample_type.map(lambda s: SAMPLE_NAME.get(s, s))
    t = t[["typ_preparatu", "typ_materialu", "pacjentki", "kafelki"]]
    print(f"\n--- Skład materiału: {label} ---")
    print(t.to_string(index=False))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=os.path.join(OUTPUT_DIR, "wsi_processed_mapping.csv"))
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "wsi_processed_mapping_FILTERED.csv"))
    ap.add_argument("--report", default=os.path.join(OUTPUT_DIR, "table_material_composition.csv"))
    ap.add_argument("--only-dx", action="store_true",
                    help="Zachowaj wyłącznie preparaty diagnostyczne FFPE (DX). "
                         "Znacznie redukuje kohortę.")
    ap.add_argument("--keep-normal", action="store_true",
                    help="NIE odrzucaj tkanki prawidłowej (odradzane)")
    ap.add_argument("--keep-metastatic", action="store_true",
                    help="Zachowaj przerzuty (kod 06). Status receptorów przerzutu "
                         "może różnić się od guza pierwotnego.")
    ap.add_argument("--min-patches", type=int, default=10,
                    help="Odrzuć pacjentki z mniejszą liczbą kafelków po filtrowaniu")
    args = ap.parse_args()

    for pth in (args.out, args.report):
        d = os.path.dirname(os.path.abspath(pth))
        if d:
            os.makedirs(d, exist_ok=True)

    print("=" * 78)
    print("FILTROWANIE PREPARATÓW WG BARCODE'U TCGA")
    print("=" * 78)

    if not os.path.exists(args.mapping):
        raise SystemExit(f"Brak {args.mapping}")
    m = pd.read_csv(args.mapping)

    src = "wsi_source" if "wsi_source" in m.columns else "original_path"
    if src not in m.columns:
        raise SystemExit("Plik mapowania nie zawiera ścieżki do preparatu źródłowego "
                         "(kolumna 'wsi_source' ani 'original_path').")

    m[["pid_bc", "sample_type", "vial", "portion", "slide_type"]] = m[src].apply(parse_barcode)

    bad = m.pid_bc.isna().sum()
    if bad:
        print(f"\n  !! Nie rozpoznano barcode'u dla {bad} kafelków — zostaną odrzucone.")
        m = m.dropna(subset=["pid_bc"])

    # Kontrola integralności: identyfikator z nazwy pliku musi zgadzać się z kolumną
    mism = (m.pid_bc != m.patient_id_norm.str.upper()).sum()
    print(f"\n--- Kontrola przypisania pacjent ↔ preparat ---")
    print(f"  Niezgodności identyfikatora: {mism} z {len(m)}")
    if mism:
        print("  !! BŁĄD KRYTYCZNY: kafelki przypisane do niewłaściwych pacjentek.")
        ex = m[m.pid_bc != m.patient_id_norm.str.upper()].head(3)
        print(ex[["patient_id_norm", "pid_bc", src]].to_string(index=False))
        raise SystemExit(1)
    print("  OK — przypisanie poprawne.")

    m["frozen"] = m.slide_type.isin(FROZEN)
    before = composition_table(m, "PRZED filtrowaniem")
    n0_pat, n0_til = m.patient_id_norm.nunique(), len(m)

    keep = pd.Series(True, index=m.index)
    reasons = []

    if not args.keep_normal:
        drop = m.sample_type == "11"
        keep &= ~drop
        reasons.append(("tkanka prawidłowa (kod 11)", int(drop.sum())))
    if not args.keep_metastatic:
        drop = m.sample_type.isin(["06", "07"])
        keep &= ~drop
        reasons.append(("przerzut (kod 06/07)", int(drop.sum())))
    if args.only_dx:
        drop = m.slide_type != "DX"
        keep &= ~drop
        reasons.append(("preparat mrożony (nie-DX)", int(drop.sum())))

    print("\n--- Odrzucone kafelki ---")
    for name, n in reasons:
        print(f"  {name:32s} {n:6d}")

    f = m[keep].copy()

    counts = f.groupby("patient_id_norm").size()
    too_few = counts[counts < args.min_patches].index
    if len(too_few):
        print(f"  {'poniżej progu ' + str(args.min_patches) + ' kafelków':32s} "
              f"{int(counts[counts < args.min_patches].sum()):6d}"
              f"  ({len(too_few)} pacjentek)")
        f = f[~f.patient_id_norm.isin(too_few)]

    composition_table(f, "PO filtrowaniu")

    lost = set(m.patient_id_norm) - set(f.patient_id_norm)
    print("\n" + "=" * 78)
    print("BILANS")
    print("=" * 78)
    print(f"  Pacjentki : {n0_pat:5d}  ->  {f.patient_id_norm.nunique():5d}"
          f"   (utracono {len(lost)})")
    print(f"  Kafelki   : {n0_til:5d}  ->  {len(f):5d}"
          f"   ({len(f)/n0_til*100:.1f}% zachowanych)")

    per = f.groupby("patient_id_norm").size()
    print(f"  Kafelków na pacjentkę: mediana {per.median():.0f}, "
          f"min {per.min()}, maks {per.max()}")
    if (per < 40).sum():
        print(f"  !! {(per < 40).sum()} pacjentek ma <40 kafelków — przy "
              f"--patches-per-patient 40 zostaną użyte wszystkie dostępne.")

    if lost:
        pd.Series(sorted(lost), name="patient_id").to_csv(
            args.out.replace(".csv", "_dropped_patients.csv"), index=False)

    out_cols = [c for c in m.columns if c not in
                ("pid_bc", "vial", "portion", "frozen")]
    f[out_cols].to_csv(args.out, index=False)

    before.to_csv(args.report, index=False)

    print("\n" + "=" * 78)
    print("ZAPISANO")
    print("=" * 78)
    print(f"  {args.out}")
    print(f"  {args.report}   (tabela składu materiału do publikacji)")
    if lost:
        print(f"  {args.out.replace('.csv', '_dropped_patients.csv')}")
    print("\n  NASTĘPNE KROKI:")
    print(f"    python step1_build_dataset.py --mapping \"{args.out}\"")
    print(f"    python step2_precompute_embeddings.py --mapping \"{args.out}\"")
    print("    python step3_train_eval_cv.py --target er --deterministic --seeds 3 \\")
    print("           --arms tab img concat concat_wide cross cross_a2b "
          "--tag final --save-preds")
    print("=" * 78)


if __name__ == "__main__":
    main()
