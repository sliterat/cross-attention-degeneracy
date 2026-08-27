# step3_train_eval_cv.py  (wersja 2)
"""
Trening i ewaluacja na buforowanych reprezentacjach obrazu.

ZMIANY WZGLEDEM WERSJI 1 - i powody:

  1. WEWNETRZNY ZBIOR WALIDACYJNY
     Wersja 1 miala EarlyStopping("loss", ...) - monitorowala strate TRENINGOWA,
     ktora prawie zawsze spada, wiec zatrzymanie nigdy nie nastepowalo i kazdy
     model przechodzil pelne 40 epok bez regularyzacji. Porownywano wiec stopnie
     przeuczenia, nie architektury.
     Teraz: wydzielony zbior walidacyjny NA POZIOMIE PACJENTA wewnatrz foldu
     treningowego, EarlyStopping na val_loss z restore_best_weights.

  2. MODALITY DROPOUT
     Diagnoza z wersji 1: korelacja wynikow cross z tab = 0.915, z img = 0.438.
     Fuzja zapadala sie na galaz tabelaryczna - polaczenie rezydualne w bloku
     uwagi (ln1(q + mha(q,kv,kv))) przepuszcza zapytanie tabelaryczne prosto na
     wyjscie, wiec model moze wyzerowac wagi uwagi i nic nie stracic.
     Teraz: losowe zerowanie calej modalnosci per-probka podczas treningu
     (Neverova i wsp., ModDrop, 2016). Zamyka droge na skroty.

  3. RAMIE KONTROLNE concat_wide
     cross ma ~449 tys. parametrow, concat ~184 tys. Bez zrownania budzetu nie
     da sie odroznic efektu MECHANIZMU od efektu POJEMNOSCI.
     Teraz: concat_wide o liczbie parametrow dopasowanej do cross.

  4. DIAGNOSTYKA POLEGANIA NA MODALNOSCI
     Po treningu liczone jest AUC przy przetasowanych wejsciach obrazu
     (i osobno tabelarycznych) miedzy pacjentami. Spadek AUC = miara faktycznego
     wykorzystania danej modalnosci. Zero spadku = modalnosc ignorowana.

Usage:
    python step3_train_eval_cv.py --target er
    python step3_train_eval_cv.py --target er --group-by-center
    python step3_train_eval_cv.py --target er --resume
    python step3_train_eval_cv.py --target er --summary-only
    python step3_train_eval_cv.py --target er --moddrop 0.0 --tag nomoddrop
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# --deterministic musi zadzialac PRZED importem TensorFlow: obie zmienne
# srodowiskowe sa odczytywane w trakcie inicjalizacji biblioteki, wiec
# ustawienie ich pozniej (np. w main()) nie ma zadnego efektu. Stad reczne
# przeszukanie argv zamiast argparse.
if "--deterministic" in sys.argv:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     StratifiedGroupKFold, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\TCGA_Data\Pipeline_Output"


# Warianty uwagi skrosnej rozkladajace architekture na czynniki:
#   bidir=True  - oba kierunki (A->B oraz B->A)
#   bidir=False - tylko A->B (kierunek B->A jest zdegenerowany, patrz nizej)
#   q_res=True  - blok A->B ma polaczenie rezydualne na zapytaniu: ln1(q + mha(...))
#   q_res=False - skrot usuniety: ln1(mha(...))
#
# DLACZEGO B->A JEST ZDEGENEROWANY:
# W tym kierunku kluczem i wartoscia jest POJEDYNCZY wektor tabelaryczny.
# Softmax po jednym elemencie wynosi zawsze 1.0, wiec wyjscie uwagi rowna sie
# dokladnie tej wartosci, niezaleznie od zapytania obrazowego. Zweryfikowane
# empirycznie: dla dwoch calkowicie roznych obrazow wyjscia sa identyczne
# co do bitu. Blok nie realizuje zadnej interakcji miedzymodalnej.
#
# DLACZEGO SKROT REZYDUALNY MA ZNACZENIE:
# W A->B zapytaniem jest wektor tabelaryczny. Skrot ln1(q + ...) przepuszcza go
# wprost na wyjscie, wiec model moze wyzerowac wagi uwagi i nie stracic sygnalu
# tabelarycznego. Obraz staje sie opcjonalny.
CROSS_VARIANTS = {
    "cross":           dict(bidir=True,  q_res=True),    # architektura wyjsciowa
    "cross_a2b":       dict(bidir=False, q_res=True),    # bez zdegenerowanego kierunku
    "cross_nores":     dict(bidir=True,  q_res=False),   # bez skrotu na zapytaniu
    "cross_a2b_nores": dict(bidir=False, q_res=False),   # obie poprawki
}

ALL_ARMS = ["tab", "img", "concat", "concat_wide"] + list(CROSS_VARIANTS)
MULTIMODAL = {"concat", "concat_wide"} | set(CROSS_VARIANTS)


# ---------------------------------------------------------------------------
# Modality dropout
# ---------------------------------------------------------------------------

class ModalityDropout(layers.Layer):
    """
    Zeruje CALA modalnosc dla losowo wybranych probek podczas treningu.

    Rozni sie od zwyklego Dropout tym, ze maska jest wspolna dla wszystkich
    wymiarow danej probki - modalnosc znika w calosci, a nie czesciowo.
    Model nie moze wiec polegac wylacznie na jednej galezi.
    """

    def __init__(self, rate=0.3, **kw):
        super().__init__(**kw)
        self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate <= 0.0:
            return x
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        keep = tf.cast(tf.random.uniform(shape) >= self.rate, x.dtype)
        return x * keep

    def get_config(self):
        cfg = super().get_config()
        cfg.update(rate=self.rate)
        return cfg


# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------

def mlp_head(x, n_out, width=64, dropout=0.3):
    x = layers.Dense(width, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    act = "sigmoid" if n_out == 1 else "softmax"
    return layers.Dense(n_out, activation=act)(x)


def cross_attention_block(d_model, num_heads, dropout, name, query_residual=True):
    mha = layers.MultiHeadAttention(num_heads=num_heads,
                                    key_dim=max(1, d_model // num_heads),
                                    dropout=dropout, name=f"{name}_mha")
    ln1 = layers.LayerNormalization(name=f"{name}_ln1")
    ffn = keras.Sequential([layers.Dense(d_model * 2, activation="relu"),
                            layers.Dense(d_model),
                            layers.Dropout(dropout)], name=f"{name}_ffn")
    ln2 = layers.LayerNormalization(name=f"{name}_ln2")

    def apply(q, kv):
        att = mha(query=q, key=kv, value=kv)
        h = ln1(q + att) if query_residual else ln1(att)
        return ln2(h + ffn(h))
    return apply


def build_model(arm, n_tab, n_tok, d_img, n_out,
                d_model=128, num_heads=4, dropout=0.2, moddrop=0.3):
    tab_in = keras.Input((n_tab,), name="tabular")
    img_in = keras.Input((n_tok, d_img), name="image_tokens")

    # ModDrop tylko dla ramion multimodalnych - w ramionach jednomodalnych
    # zerowanie jedynej modalnosci pozostawiloby model bez zadnego wejscia.
    t = ModalityDropout(moddrop, name="moddrop_tab")(tab_in) if arm in MULTIMODAL else tab_in
    v = ModalityDropout(moddrop, name="moddrop_img")(img_in) if arm in MULTIMODAL else img_in

    if arm == "tab":
        h = layers.Dense(d_model, activation="relu")(t)
        inputs = [tab_in]

    elif arm == "img":
        z = layers.Dense(d_model)(v)
        h = layers.GlobalAveragePooling1D()(z)
        inputs = [img_in]

    elif arm == "concat":
        a = layers.Dense(d_model, activation="relu")(t)
        b = layers.GlobalAveragePooling1D()(layers.Dense(d_model)(v))
        h = layers.Concatenate()([a, b])
        inputs = [tab_in, img_in]

    elif arm == "concat_wide":
        # RAMIE KONTROLNE: ta sama fuzja co concat, budzet parametrow jak cross.
        # Jesli concat_wide dorownuje cross, przewaga cross bierze sie z pojemnosci,
        # a nie z mechanizmu uwagi.
        a = layers.Dense(256, activation="relu")(t)
        b = layers.GlobalAveragePooling1D()(layers.Dense(256)(v))
        h = layers.Concatenate()([a, b])
        h = layers.Dense(128, activation="relu")(h)
        h = layers.Dropout(dropout)(h)
        inputs = [tab_in, img_in]

    elif arm in CROSS_VARIANTS:
        cfg = CROSS_VARIANTS[arm]
        q = layers.Reshape((1, d_model))(layers.Dense(d_model)(t))
        kv = layers.Dense(d_model)(v)

        a2b = cross_attention_block(d_model, num_heads, dropout, "a2b",
                                    query_residual=cfg["q_res"])
        pooled_a = layers.GlobalAveragePooling1D()(a2b(q, kv))

        if cfg["bidir"]:
            b2a = cross_attention_block(d_model, max(1, num_heads // 2), dropout, "b2a")
            pooled_b = layers.GlobalAveragePooling1D()(b2a(kv, q))
            h = layers.Concatenate()([pooled_a, pooled_b])
        else:
            # kierunek A->B niesie wylacznie odczyt obrazu sterowany kontekstem
            # klinicznym; zeby wariant nie stracil dostepu do danych tabelarycznych,
            # dolaczamy je jawnie - inaczej porownanie z 'cross' bylo by nieuczciwe
            h = layers.Concatenate()([pooled_a,
                                      layers.Dense(d_model, activation="relu")(t)])
        inputs = [tab_in, img_in]
    else:
        raise ValueError(arm)

    out = mlp_head(h, n_out)
    return keras.Model(inputs, out, name=f"arm_{arm}")


def arm_inputs(arm, Xt, Xi):
    if arm == "tab":
        return Xt
    if arm == "img":
        return Xi
    return [Xt, Xi]


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------

def load_all(clinical_csv, features_csv, cache_dir, target):
    df = pd.read_csv(clinical_csv)
    feat_cols = [c for c in pd.read_csv(features_csv)["feature"] if c in df.columns]

    if target == "subtype":
        df = df[df.molecular_subtype != -1].copy()
        y_col, n_out = "molecular_subtype", 4
    elif target == "er":
        df = df.dropna(subset=["y_er"]).copy()
        df["y_er"] = df["y_er"].astype(int)
        y_col, n_out = "y_er", 1
    elif target == "tnbc":
        df = df.dropna(subset=["y_tnbc"]).copy()
        df["y_tnbc"] = df["y_tnbc"].astype(int)
        y_col, n_out = "y_tnbc", 1
    else:
        raise ValueError(target)

    index = pd.read_csv(os.path.join(cache_dir, "embeddings_index.csv"))
    emb = np.load(os.path.join(cache_dir, "embeddings.npy"), mmap_mode="r")

    keep = set(df.patient_id_norm) & set(index.patient_id)
    df = df[df.patient_id_norm.isin(keep)].reset_index(drop=True)
    pos = {p: g.index.values for p, g in index.groupby("patient_id")}
    return df, feat_cols, y_col, n_out, emb, pos


def make_arrays(df, ids, feat_cols, y_col, emb, pos, scaler):
    sub = df[df.patient_id_norm.isin(ids)]
    Xt, rows, y, pid = [], [], [], []
    for _, r in sub.iterrows():
        idxs = pos.get(r.patient_id_norm)
        if idxs is None or len(idxs) == 0:
            continue
        t = r[feat_cols].values.astype(np.float32)
        for ri in idxs:
            Xt.append(t); rows.append(ri); y.append(r[y_col]); pid.append(r.patient_id_norm)
    Xt = scaler.transform(np.array(Xt, np.float32))
    Xi = np.asarray(emb[np.array(rows)], np.float32)
    return Xt, Xi, np.array(y), np.array(pid)


def aggregate_to_patient(proba, pids, y_patch, return_ids=False):
    out_p, out_y, out_id = [], [], []
    for p in pd.unique(pids):
        m = pids == p
        out_p.append(proba[m].mean(axis=0))
        out_y.append(y_patch[m][0])
        out_id.append(p)
    if return_ids:
        return np.array(out_p), np.array(out_y), np.array(out_id)
    return np.array(out_p), np.array(out_y)


# ---------------------------------------------------------------------------
# Metryki
# ---------------------------------------------------------------------------

def score(y, proba, n_out):
    if n_out == 1:
        p = proba.ravel()
        return {"auc": roc_auc_score(y, p),
                "ap": average_precision_score(y, p),
                "acc": float(((p > .5).astype(int) == y).mean()),
                "f1": f1_score(y, (p > .5).astype(int), zero_division=0)}
    pred = proba.argmax(1)
    d = {"acc": float((pred == y).mean()),
         "f1_macro": f1_score(y, pred, average="macro", zero_division=0)}
    try:
        d["auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    except ValueError:
        d["auc"] = np.nan
    return d


def auc_only(y, proba, n_out):
    try:
        return score(y, proba, n_out)["auc"]
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", default=os.path.join(OUTPUT_DIR, "dataset_FIXED.csv"))
    ap.add_argument("--features", default=os.path.join(OUTPUT_DIR, "feature_columns.csv"))
    ap.add_argument("--cache", default=os.path.join(OUTPUT_DIR, "cache"))
    ap.add_argument("--outdir", default=os.path.join(OUTPUT_DIR, "cv_results"))
    ap.add_argument("--target", default="er", choices=["er", "tnbc", "subtype"])
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--moddrop", type=float, default=0.3)
    ap.add_argument("--monitor", default="val_auc",
                    help="Metryka dla EarlyStopping (val_auc lub val_loss)")
    ap.add_argument("--monitor-mode", default="max", choices=["max", "min"])
    ap.add_argument("--group-by-center", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Liczba ziaren inicjalizacji na kazdy fold. Wariancja "
                         "inicjalizacji jest przy malych kohortach porownywalna "
                         "z badanymi efektami, wiec wymaga usredniania.")
    ap.add_argument("--save-preds", action="store_true",
                    help="Zapisuje predykcje na poziomie pacjenta (potrzebne do "
                         "analizy fuzji poznej i zgodnosci modalnosci)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Pelna odtwarzalnosc bit-w-bit (wolniejsze o ok. 10-30%)")
    args = ap.parse_args()

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
        # Wielowatkowosc CPU jest osobnym zrodlem niedeterminizmu: kolejnosc
        # redukcji zmiennoprzecinkowych zalezy od szeregowania watkow.
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    os.makedirs(args.outdir, exist_ok=True)
    suffix = args.target + ("_bycenter" if args.group_by_center else "") + \
             (f"_{args.tag}" if args.tag else "")
    res_path = os.path.join(args.outdir, f"cv_folds_{suffix}.csv")

    print("=" * 78)
    print(f"WALIDACJA KRZYZOWA v2  |  cel: {args.target}  |  {args.repeats}x{args.folds}")
    print(f"  ModDrop={args.moddrop}  val_frac={args.val_frac}  patience={args.patience}")
    print(f"  determinizm: {'WLACZONY' if args.deterministic else 'wylaczony'}"
          + ("" if args.deterministic else
             "  (powtorzenia moga dac rozne wyniki na poziomie foldu)"))
    print("=" * 78)

    df, feat_cols, y_col, n_out, emb, pos = load_all(
        args.clinical, args.features, args.cache, args.target)

    pids = df.patient_id_norm.values
    y_pat = df[y_col].values.astype(int)
    centers = pd.Series(pids).str.split("-").str[1].values

    print(f"\n  Pacjentow: {len(pids)}  |  cech: {len(feat_cols)}  |  osrodkow: {len(set(centers))}")
    if args.group_by_center:
        top = pd.Series(centers).value_counts()
        print(f"  Najwiekszy osrodek: {top.index[0]} = {top.iloc[0]} pacjentow "
              f"({top.iloc[0]/len(centers)*100:.0f}% kohorty)")
    print(f"  Rozklad celu: {dict(pd.Series(y_pat).value_counts().sort_index())}")
    print(f"  Bufor obrazu: {emb.shape}")

    print("\n--- Budzet parametrow (kontrola pojemnosci) ---")
    for a in args.arms:
        mm = build_model(a, len(feat_cols), emb.shape[1], emb.shape[2], n_out,
                         moddrop=args.moddrop)
        print(f"  {a:12s} {mm.count_params():>9,d}")
        keras.backend.clear_session()

    rows, done = [], set()
    if (args.resume or args.summary_only) and os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        rows = prev.to_dict("records")
        sd = prev["seed"] if "seed" in prev.columns else pd.Series(0, index=prev.index)
        done = set(zip(prev.rep, prev.fold, prev.arm, sd))
        print(f"\n  Wznowienie: {len(rows)} gotowych wynikow")

    total = args.repeats * args.folds * len(args.arms) * args.seeds

    schema = [None]

    preds_path = os.path.join(args.outdir, f"cv_preds_{suffix}.csv")
    if args.save_preds and not args.resume and os.path.exists(preds_path):
        os.replace(preds_path, preds_path.replace(".csv", "_OLD.csv"))

    def flush_preds(rows_pred):
        pd.DataFrame(rows_pred).to_csv(
            preds_path, mode="a",
            header=not os.path.exists(preds_path), index=False)

    def flush(rec):
        # Zabezpieczenie: jesli istniejacy plik ma inny zestaw kolumn (np. pochodzi
        # z wczesniejszej wersji skryptu), zostaje zarchiwizowany zamiast sklejony.
        cols = sorted(rec.keys())
        if schema[0] is None:
            schema[0] = cols
            if os.path.exists(res_path):
                try:
                    old_cols = sorted(pd.read_csv(res_path, nrows=1).columns)
                except Exception:
                    old_cols = None
                if old_cols != cols:
                    bak = res_path.replace(".csv", "_OLD.csv")
                    os.replace(res_path, bak)
                    print(f"  !! Istniejacy plik mial inny format - przeniesiony do {bak}")
        pd.DataFrame([rec]).to_csv(res_path, mode="a",
                                   header=not os.path.exists(res_path), index=False)

    if args.summary_only:
        if not rows:
            print(f"\n  !! Brak {res_path}")
            sys.exit(1)
        summarize(pd.DataFrame(rows), args, total)
        return

    for rep in range(args.repeats):
        if args.group_by_center:
            # StratifiedGroupKFold zamiast GroupKFold: ten drugi nie stratyfikuje,
            # a rozklad osrodkow jest skrajnie nierowny (najwiekszy = 20% kohorty),
            # przez co jeden fold moze byc pojedynczym szpitalem o skosnym
            # rozkladzie klas. Dodatkowo StratifiedGroupKFold przyjmuje shuffle,
            # wiec powtorzenia maja sens i statystyka staje sie mozliwa.
            splitter = StratifiedGroupKFold(
                n_splits=args.folds, shuffle=True,
                random_state=rep).split(pids, y_pat, groups=centers)
        else:
            splitter = StratifiedKFold(args.folds, shuffle=True,
                                       random_state=rep).split(pids, y_pat)

        for fold, (tr, te) in enumerate(splitter):
            ids_full, ids_te = pids[tr], pids[te]
            y_full = y_pat[tr]

            strat = y_full if pd.Series(y_full).value_counts().min() >= 2 else None
            ids_tr, ids_val = train_test_split(
                ids_full, test_size=args.val_frac,
                random_state=rep * 100 + fold, stratify=strat)

            block = df[df.patient_id_norm.isin(ids_tr)][feat_cols].astype(np.float32)
            medians = block.median()
            df_imp = df.copy()
            df_imp[feat_cols] = df_imp[feat_cols].astype(np.float32).fillna(medians)
            scaler = StandardScaler().fit(block.fillna(medians).values)

            def A(ids):
                return make_arrays(df_imp, ids, feat_cols, y_col, emb, pos, scaler)

            Xt_tr, Xi_tr, y_tr, _ = A(ids_tr)
            Xt_va, Xi_va, y_va, _ = A(ids_val)
            Xt_te, Xi_te, y_te, p_te = A(ids_te)

            if n_out == 1:
                n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
                cw = {0: len(y_tr) / (2 * max(n0, 1)), 1: len(y_tr) / (2 * max(n1, 1))}
                y_tr_fit, y_va_fit, loss = y_tr, y_va, "binary_crossentropy"
            else:
                cnt = np.bincount(y_tr, minlength=n_out)
                cw = {i: len(y_tr) / (n_out * max(c, 1)) for i, c in enumerate(cnt)}
                y_tr_fit = keras.utils.to_categorical(y_tr, n_out)
                y_va_fit = keras.utils.to_categorical(y_va, n_out)
                loss = "categorical_crossentropy"

            if args.group_by_center and rep == 0:
                nc = len(set(centers[te]))
                print(f"    fold{fold}: n={len(ids_te)} osrodkow={nc}"
                      f" HR-={int((y_pat[te]==0).sum())} HR+={int((y_pat[te]==1).sum())}"
                      + ("  !! fold = jeden osrodek" if nc == 1 else ""))

            arm_seed = [(a, sd) for a in args.arms for sd in range(args.seeds)]
            for arm, seed_idx in arm_seed:
                if (rep, fold, arm, seed_idx) in done:
                    print(f"  rep{rep} fold{fold} s{seed_idx} {arm:16s} -- pominiete")
                    continue

                keras.backend.clear_session()
                tf.keras.utils.set_random_seed(1000 * rep + fold + 100000 * seed_idx)

                m = build_model(arm, Xt_tr.shape[1], Xi_tr.shape[1], Xi_tr.shape[2],
                                n_out, moddrop=args.moddrop)
                # AUC jako metryka monitorujaca: val_loss jest liczony BEZ wag klas,
                # wiec walczy z class_weight uzytym w treningu i zatrzymuje model
                # na 1. epoce. AUC opiera sie na rankingu, jest odporne na ta
                # niespojnosc i na niezbalansowanie klas.
                mon_metric = keras.metrics.AUC(
                    name="auc", multi_label=(n_out > 1),
                    num_labels=(n_out if n_out > 1 else None))
                m.compile(keras.optimizers.Adam(args.lr), loss=loss,
                          metrics=[mon_metric])

                hist = m.fit(
                    arm_inputs(arm, Xt_tr, Xi_tr), y_tr_fit,
                    validation_data=(arm_inputs(arm, Xt_va, Xi_va), y_va_fit),
                    epochs=args.epochs, batch_size=args.batch_size,
                    class_weight=cw, verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(
                        args.monitor, mode=args.monitor_mode,
                        patience=args.patience, min_delta=1e-4,
                        restore_best_weights=True)])

                proba = m.predict(arm_inputs(arm, Xt_te, Xi_te), verbose=0, batch_size=256)
                P, Y, ids = aggregate_to_patient(proba, p_te, y_te, return_ids=True)
                s = score(Y, P, n_out)

                if args.save_preds:
                    flush_preds([
                        dict(rep=rep, fold=fold, seed=seed_idx, arm=arm,
                             patient_id=pid, y_true=int(yv),
                             y_prob=float(pv.ravel()[0]) if n_out == 1
                                    else float(pv.max()),
                             y_pred=int(pv.ravel()[0] > .5) if n_out == 1
                                    else int(pv.argmax()))
                        for pid, yv, pv in zip(ids, Y, P)])

                rng = np.random.RandomState(0)
                if arm in MULTIMODAL:
                    perm = rng.permutation(len(Xi_te))
                    pr = m.predict([Xt_te, Xi_te[perm]], verbose=0, batch_size=256)
                    Pp, Yp = aggregate_to_patient(pr, p_te, y_te)
                    s["auc_img_shuf"] = auc_only(Yp, Pp, n_out)

                    perm = rng.permutation(len(Xt_te))
                    pr = m.predict([Xt_te[perm], Xi_te], verbose=0, batch_size=256)
                    Pp, Yp = aggregate_to_patient(pr, p_te, y_te)
                    s["auc_tab_shuf"] = auc_only(Yp, Pp, n_out)

                    s["rely_img"] = s["auc"] - s["auc_img_shuf"]
                    s["rely_tab"] = s["auc"] - s["auc_tab_shuf"]
                else:
                    s["auc_img_shuf"] = s["auc_tab_shuf"] = np.nan
                    s["rely_img"] = s["rely_tab"] = np.nan

                s["epochs_run"] = len(hist.history["loss"])
                key = args.monitor if args.monitor in hist.history else "val_loss"
                pick = np.argmax if args.monitor_mode == "max" else np.argmin
                s["best_epoch"] = int(pick(hist.history[key])) + 1
                s.update(rep=rep, fold=fold, arm=arm, seed=seed_idx, n_test=len(Y))
                rows.append(s)
                flush(s)
                print(f"  [{len(rows):3d}/{total}] rep{rep} fold{fold} s{seed_idx} {arm:16s} "
                      f"auc={s['auc']:.3f} ep={s['epochs_run']:3d}"
                      + (f" rely_img={s['rely_img']:+.3f}" if arm in MULTIMODAL else ""))

    summarize(pd.DataFrame(rows), args, total)


def summarize(res, args, total):
    print("\n" + "=" * 78)
    print("PODSUMOWANIE  (poziom pacjenta)")
    print("=" * 78)

    if len(res) < total:
        print(f"  UWAGA: przebieg niepelny ({len(res)}/{total}). Dokoncz przez --resume.")

    core = [c for c in ["auc", "ap", "acc", "f1", "f1_macro"] if c in res.columns]
    print("\n--- Metryki ---")
    print(res.groupby("arm")[core].agg(["mean", "std"]).round(3).to_string())

    if "epochs_run" in res.columns:
        print("\n--- Epoki do zatrzymania (kontrola dzialania early stopping) ---")
        print(res.groupby("arm")["epochs_run"].agg(["mean", "min", "max"]).round(1).to_string())
        if (res["epochs_run"] >= args.epochs).all():
            print("  !! Wszystkie modele wyczerpaly limit epok - zwieksz --epochs.")
    if "best_epoch" in res.columns:
        print("\n--- Najlepsza epoka (kontrola nie/przeuczenia) ---")
        print(res.groupby("arm")["best_epoch"].agg(["mean", "median", "max"]).round(1).to_string())
        frac = (res["best_epoch"] <= 1).mean()
        if frac > 0.3:
            print(f"  !! {frac*100:.0f}% modeli zatrzymuje sie na 1. epoce - modele niedouczone.")
            print("     Sprobuj --lr 3e-4 lub --monitor val_auc.")

    if "rely_img" in res.columns and res["rely_img"].notna().any():
        print("\n--- Poleganie na modalnosci (spadek AUC po przetasowaniu wejscia) ---")
        r = res.dropna(subset=["rely_img"]).groupby("arm")[["rely_img", "rely_tab"]]
        print(r.agg(["mean", "std"]).round(3).to_string())
        print("\n  rely_img ~ 0    -> obraz praktycznie ignorowany (zapadniecie fuzji)")
        print("  rely_img > 0.05 -> obraz realnie wykorzystywany")

    if len(res.arm.unique()) > 1 and "auc" in res.columns:
        from scipy.stats import wilcoxon
        P = res.pivot_table(index=["rep", "fold"], columns="arm", values="auc")
        arms = [a for a in ALL_ARMS if a in P.columns]
        pairs = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                x, y = P[a].dropna(), P[b].dropna()
                idx = x.index.intersection(y.index)
                if len(idx) < 5:
                    continue
                try:
                    _, p = wilcoxon(x[idx], y[idx])
                except Exception:
                    continue
                pairs.append((f"{a} vs {b}", x[idx].mean() - y[idx].mean(), p))
        pairs.sort(key=lambda t: t[2])
        n = len(pairs)
        n_folds = len(P)
        print("\n--- Porownania parami (Wilcoxon, poprawka Holma) ---")
        if n_folds < 8:
            print(f"  !! Tylko {n_folds} foldow. Minimalne osiagalne p = {2**-n_folds*2:.4f};")
            print(f"     zaden wynik nie moze byc istotny. Zwieksz --repeats.")
        for k, (nm, dm, p) in enumerate(pairs):
            holm = min(1.0, p * (n - k))
            print(f"  {nm:26s} dAUC={dm:+.3f}  p={p:.4f}  p_Holm={holm:.4f}"
                  f"  {'*' if holm < 0.05 else ''}")

        print("\n--- Zrodla zmiennosci AUC ---")
        sd_fold = P.mean(axis=1).std(ddof=1)
        sd_arm = P.mean(axis=0).std(ddof=1)
        print(f"  SD miedzy foldami       : {sd_fold:.4f}")
        print(f"  SD miedzy ramionami     : {sd_arm:.4f}")
        if "seed" in res.columns and res["seed"].nunique() > 1:
            # rozrzut miedzy ziarnami przy ustalonym foldzie i ramieniu
            within = (res.groupby(["rep", "fold", "arm"])["auc"]
                         .std(ddof=1).mean())
            print(f"  SD miedzy ziarnami      : {within:.4f}   <- ten sam fold, to samo ramie")
            if within > sd_arm:
                print("  !! Wariancja inicjalizacji PRZEKRACZA roznice miedzy architekturami.")
                print("     Wnioski z pojedynczego ziarna sa nieuprawnione.")
        if sd_fold > 0:
            print(f"  stosunek war. ramie/fold: {(sd_arm/sd_fold)**2:.4f}")

    print("=" * 78)


if __name__ == "__main__":
    main()
