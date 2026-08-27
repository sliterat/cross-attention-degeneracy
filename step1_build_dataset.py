# step1_build_dataset.py
"""
KROK 1: Budowa zbioru danych z pliku klinicznego TCGA (wersja 2).

Zastepuje: 1_extract_clinical_tcga.py, xml_parser.py, 3_split_data_subtypes.py
oraz brakujacy create_molecular_subtypes.py.

Dlaczego czytamy TXT zamiast XML:
  - plik TXT zawiera her2_fish_status, ktorego parser XML nie pobieral;
    FISH rozstrzyga przypadki HER2 equivocal zgodnie z wytycznymi ASCO/CAP
    i powieksza kohorte oznaczalna o okolo 40%
  - zawiera tez pola nieobecne w sciezce XML: menopause_status, ajcc_tumor_pathologic_pt,
    margin_status, anatomic_neoplasm_subdivision
  - jest jednym plikiem zamiast 1174 - brak ryzyka bledow scalania

UWAGA TECHNICZNA: plik ma TRZY wiersze naglowka (nazwy krotkie, nazwy dlugie CDE,
identyfikatory CDE). Wiersze 2 i 3 nalezy pominac - poprzedni pipeline tego nie
robil i wczytywal je jako dane.

Usage:
    python step1_build_dataset.py
    python step1_build_dataset.py --task subtype
    python step1_build_dataset.py --no-fish        # ablacja: bez ratunku FISH
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

try:
    from config import OUTPUT_DIR, CLINICAL_DIR
except ImportError:
    OUTPUT_DIR = r"D:\TCGA_Data\Pipeline_Output"
    CLINICAL_DIR = r"D:\TCGA_Data\Clinical"

MISSING_TOKENS = {
    "[Not Available]", "[Not Applicable]", "[Not Evaluated]", "[Unknown]",
    "[Discrepancy]", "[Completed]", "[Pending]", "NA", "N/A", "", "nan",
}

SUBTYPE_NAMES = {0: "Luminal A", 1: "Luminal B", 2: "HER2-enriched",
                 3: "Triple-negative", -1: "Nieoznaczalny"}

VALID_STATUS = {"positive", "negative"}


# ---------------------------------------------------------------------------
# Wczytanie
# ---------------------------------------------------------------------------

def read_tcga_txt(path):
    """Wczytuje plik TCGA pomijajac dwa dodatkowe wiersze naglowka."""
    df = pd.read_csv(path, sep="\t", low_memory=False, skiprows=[1, 2])
    df = df.replace(list(MISSING_TOKENS), np.nan)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace(list(MISSING_TOKENS), np.nan)
    df["patient_id_norm"] = df["bcr_patient_barcode"].astype(str).str.upper().str.strip()
    return df


def norm_status(v):
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v.startswith("pos"):
        return "positive"
    if v.startswith("neg"):
        return "negative"
    if v.startswith("equiv"):
        return "equivocal"
    return None


# ---------------------------------------------------------------------------
# Etykiety
# ---------------------------------------------------------------------------

def build_receptor_status(df, use_fish=True):
    df = df.copy()
    df["ER"] = df["er_status_by_ihc"].map(norm_status)
    df["PR"] = df["pr_status_by_ihc"].map(norm_status)
    df["HER2_ihc"] = df["her2_status_by_ihc"].map(norm_status)
    df["HER2_fish"] = df["her2_fish_status"].map(norm_status) if use_fish else None

    def resolve(r):
        # ASCO/CAP: wynik FISH jest rozstrzygajacy wobec IHC
        if use_fish and r["HER2_fish"] in VALID_STATUS:
            return r["HER2_fish"]
        if r["HER2_ihc"] in VALID_STATUS:
            return r["HER2_ihc"]
        return None

    df["HER2"] = df.apply(resolve, axis=1)
    df["her2_resolved_by"] = np.where(
        use_fish & df["HER2_fish"].isin(VALID_STATUS), "FISH",
        np.where(df["HER2_ihc"].isin(VALID_STATUS), "IHC", "brak"))
    return df


def assign_subtype(r):
    """Reguła St. Gallen. -1 gdy ktorykolwiek receptor nierozstrzygniety."""
    er, pr, h = r["ER"], r["PR"], r["HER2"]
    if not (er in VALID_STATUS and pr in VALID_STATUS and h in VALID_STATUS):
        return -1
    if h == "positive":
        return 1 if (er == "positive" or pr == "positive") else 2
    if er == "positive" or pr == "positive":
        return 0
    return 3


# ---------------------------------------------------------------------------
# Cechy
# ---------------------------------------------------------------------------

def parse_T(v):
    """T1a..T4d -> 1..4; TX -> NaN. Zastepuje nieobecny tumor_size_mm."""
    if not isinstance(v, str):
        return np.nan
    v = v.strip().upper()
    for k in ("T1", "T2", "T3", "T4"):
        if v.startswith(k):
            return float(k[1])
    return np.nan


def parse_N(v):
    """N0..N3 -> 0..3; NX -> NaN."""
    if not isinstance(v, str):
        return np.nan
    v = v.strip().upper()
    for k in ("N0", "N1", "N2", "N3"):
        if v.startswith(k):
            return float(k[1])
    return np.nan


def parse_menopause(v):
    if not isinstance(v, str):
        return np.nan
    v = v.lower()
    if v.startswith("pre"):
        return 0.0
    if v.startswith("peri"):
        return 1.0
    if v.startswith("post"):
        return 2.0
    return np.nan


def build_features(df):
    """
    Buduje przestrzen cech BEZ zmiennych definiujacych etykiete
    i BEZ zmiennych post-diagnostycznych.

    Swiadomie NIEUZYTE:
      er/pr/her2*                  - definiuja etykiete (wyciek)
      vital_status, death_days_to  - zdarzenia po diagnozie
      last_contact_days_to         - jw.
      tumor_grade                  - NIE ISTNIEJE w TCGA-BRCA
    """
    out = pd.DataFrame({"patient_id_norm": df["patient_id_norm"]})

    out["age"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
    out["menopause"] = df["menopause_status"].map(parse_menopause)
    out["T_stage"] = df["ajcc_tumor_pathologic_pt"].map(parse_T)
    out["N_stage"] = df["ajcc_nodes_pathologic_pn"].map(parse_N)
    out["nodes_examined"] = pd.to_numeric(df["lymph_nodes_examined_count"], errors="coerce")
    out["nodes_positive_he"] = pd.to_numeric(df["lymph_nodes_examined_he_count"], errors="coerce")
    out["neoadjuvant"] = (df["history_neoadjuvant_treatment"].astype(str)
                          .str.lower().eq("yes").astype(float))

    # kategorie: one-hot z jawnym wskaznikiem braku danych
    for col, prefix, top in [("race", "race", 3),
                             ("histological_type", "hist", 5),
                             ("margin_status", "margin", 3)]:
        s = df[col].fillna("__missing__")
        keep = [v for v in s.value_counts().index[:top] if v != "__missing__"]
        for v in keep:
            name = f"{prefix}_" + "".join(ch if ch.isalnum() else "_" for ch in str(v))[:28]
            out[name] = (s == v).astype(float)
        out[f"{prefix}_missing"] = (s == "__missing__").astype(float)

    # laterality z anatomic_neoplasm_subdivision
    side = df["anatomic_neoplasm_subdivision"].fillna("")
    out["side_left"] = side.str.contains("Left", case=False).astype(float)

    # mediana liczona PO podziale - tu tylko wskazniki braku
    for c in ["age", "menopause", "T_stage", "N_stage", "nodes_examined", "nodes_positive_he"]:
        out[f"{c}_missing"] = out[c].isna().astype(float)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default=os.path.join(
        CLINICAL_DIR, "nationwidechildrens.org_clinical_patient_brca.txt"))
    ap.add_argument("--mapping", default=os.path.join(OUTPUT_DIR, "wsi_processed_mapping.csv"))
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--task", default="er", choices=["er", "tnbc", "subtype"])
    ap.add_argument("--no-fish", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("KROK 1: BUDOWA ZBIORU DANYCH")
    print("=" * 74)

    if not os.path.exists(args.txt):
        print(f"\n!! Nie znaleziono {args.txt}")
        sys.exit(1)

    raw = read_tcga_txt(args.txt)
    print(f"\n  Wczytano: {len(raw)} pacjentow, {raw.shape[1]} kolumn")

    raw = build_receptor_status(raw, use_fish=not args.no_fish)
    raw["molecular_subtype"] = raw.apply(assign_subtype, axis=1)

    print("\n--- Rozstrzygniecie statusu HER2 ---")
    print(raw["her2_resolved_by"].value_counts().to_string())
    if not args.no_fish:
        only_ihc = raw["HER2_ihc"].isin(VALID_STATUS).sum()
        print(f"  Zysk dzieki FISH: +{raw['HER2'].isin(VALID_STATUS).sum() - only_ihc} pacjentow")

    feats = build_features(raw)
    data = feats.merge(
        raw[["patient_id_norm", "molecular_subtype", "ER", "PR", "HER2"]],
        on="patient_id_norm", how="left")

    # cele
    det = data["molecular_subtype"] != -1
    data["y_er"] = np.where(det, data["molecular_subtype"].isin([0, 1]).astype(float), np.nan)
    data["y_tnbc"] = np.where(det, (data["molecular_subtype"] == 3).astype(float), np.nan)
    data["center"] = data["patient_id_norm"].str.split("-").str[1]

    # ograniczenie do pacjentow z obrazami
    if os.path.exists(args.mapping):
        with_img = set(pd.read_csv(args.mapping)["patient_id_norm"].astype(str).str.upper())
        before = len(data)
        data = data[data["patient_id_norm"].isin(with_img)].copy()
        print(f"\n  Filtr obrazow: {before} -> {len(data)} pacjentow")
    else:
        print(f"\n  !! Brak {args.mapping} - pomijam filtr obrazow")

    print("\n--- Rozklad podtypow (kohorta koncowa) ---")
    d = data[data["molecular_subtype"] != -1]
    for k, v in d["molecular_subtype"].value_counts().sort_index().items():
        print(f"  {SUBTYPE_NAMES[int(k)]:18s}: {v:4d} ({v/len(d)*100:5.1f}%)")
    print(f"  {'Nieoznaczalnych':18s}: {(data['molecular_subtype']==-1).sum():4d} (wykluczonych)")

    y_col = {"er": "y_er", "tnbc": "y_tnbc", "subtype": "molecular_subtype"}[args.task]
    usable = data.dropna(subset=[y_col])
    usable = usable[usable[y_col] != -1] if args.task == "subtype" else usable
    print(f"\n--- Zadanie: {args.task} ---")
    print(f"  Uzytecznych pacjentow: {len(usable)}")
    print(f"  Rozklad celu: {dict(usable[y_col].value_counts().sort_index())}")

    feature_cols = [c for c in feats.columns if c != "patient_id_norm"]
    print(f"\n  Cech predykcyjnych: {len(feature_cols)}")

    # test kontrolny wycieku
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    X = usable[feature_cols].fillna(usable[feature_cols].median()).values
    y = usable[y_col].astype(int).values
    if len(np.unique(y)) > 1:
        cv = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean()
        base = pd.Series(y).value_counts(normalize=True).max()
        print(f"\n--- Test kontrolny wycieku ---")
        print(f"  Drzewo na samych cechach, 5-fold CV: {cv:.4f}  (baseline {base:.4f})")
        print("  >>> ALARM: WYCIEK" if cv > 0.95 else "  >>> OK: brak trywialnego wycieku")

    os.makedirs(args.outdir, exist_ok=True)
    p1 = os.path.join(args.outdir, "dataset_FIXED.csv")
    p2 = os.path.join(args.outdir, "feature_columns.csv")
    data.to_csv(p1, index=False)
    pd.Series(feature_cols, name="feature").to_csv(p2, index=False)

    print("\n" + "=" * 74)
    print("ZAPISANO")
    print("=" * 74)
    print(f"  {p1}")
    print(f"  {p2}")
    print("\n  NASTEPNY KROK:  python step2_precompute_embeddings.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
