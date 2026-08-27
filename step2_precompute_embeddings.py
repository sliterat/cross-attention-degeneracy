# precompute_embeddings.py
"""
KROK JEDNORAZOWY: przelicza kafelki przez zamrozony enkoder obrazu
i zapisuje reprezentacje na dysk (memmap fp16).

Po tym kroku wszystkie eksperymenty (walidacja krzyzowa, ablacje, strojenie
hiperparametrow) dzialaja na buforze i trwaja sekundy, takze na CPU.

Wyjscie:
    embeddings.npy      memmap  (N_patches, 49, D)   float16
    embeddings_index.csv         patient_id, patch_idx, path

Szacunkowy rozmiar dla EfficientNetB0, 40 kafelkow/pacjenta, 298 pacjentow:
    11 920 x 49 x 1280 x 2B  ~=  1.5 GB

Usage:
    python precompute_embeddings.py
    python precompute_embeddings.py --encoder b0 --patches-per-patient 40
    python precompute_embeddings.py --batch-size 8        # gdy malo RAM/VRAM
    python precompute_embeddings.py --resume              # dokonczenie po przerwaniu
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\TCGA_Data\Pipeline_Output"


ENCODERS = {
    "b0": ("EfficientNetB0", 1280, "efficientnet"),
    "b4": ("EfficientNetB4", 1792, "efficientnet"),
    "r50": ("ResNet50V2", 2048, "resnet_v2"),
}


def build_encoder(kind, img_size=224):
    """
    Zamrozony enkoder zwracajacy SEKWENCJE tokenow przestrzennych (7x7 -> 49),
    a nie pojedynczy wektor po global average pooling.

    To jest warunek konieczny dzialania uwagi skrosnej: nie da sie selektywnie
    zwrocic uwagi na jeden wektor.
    """
    name, dim, module = ENCODERS[kind]
    Base = getattr(keras.applications, name)
    preprocess = getattr(keras.applications, module).preprocess_input

    base = Base(include_top=False, weights="imagenet",
                input_shape=(img_size, img_size, 3))
    base.trainable = False

    inp = keras.Input((img_size, img_size, 3))
    fmap = base(inp, training=False)                      # (B, 7, 7, C)
    seq = keras.layers.Reshape((-1, fmap.shape[-1]))(fmap)  # (B, 49, C)
    return keras.Model(inp, seq, name=f"{kind}_frozen"), preprocess, dim


def select_patches(df_map, per_patient, seed=42):
    """
    Deterministyczny wybor podzbioru kafelkow na pacjenta.

    Deterministyczny, bo caly sens buforowania znika, jesli przy kazdym
    uruchomieniu wybierany jest inny podzbior.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for pid, g in df_map.groupby("patient_id_norm", sort=True):
        g = g.sort_values("processed_path").reset_index(drop=True)
        if per_patient and len(g) > per_patient:
            idx = np.sort(rng.choice(len(g), per_patient, replace=False))
            g = g.iloc[idx]
        for j, (_, r) in enumerate(g.iterrows()):
            rows.append({"patient_id": pid, "patch_idx": j,
                         "path": r["processed_path"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=os.path.join(OUTPUT_DIR, "wsi_processed_mapping.csv"))
    ap.add_argument("--outdir", default=os.path.join(OUTPUT_DIR, "cache"))
    ap.add_argument("--encoder", default="b0", choices=list(ENCODERS))
    ap.add_argument("--patches-per-patient", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    emb_path = os.path.join(args.outdir, "embeddings.npy")
    idx_path = os.path.join(args.outdir, "embeddings_index.csv")
    done_path = os.path.join(args.outdir, "_progress.txt")

    print("=" * 74)
    print("PRZELICZANIE REPREZENTACJI OBRAZU (jednorazowo)")
    print("=" * 74)

    df_map = pd.read_csv(args.mapping)
    index = select_patches(df_map, args.patches_per_patient, args.seed)
    index.to_csv(idx_path, index=False)

    N = len(index)
    encoder, preprocess, D = build_encoder(args.encoder, args.img_size)
    n_tok = encoder.output_shape[1]

    gb = N * n_tok * D * 2 / 1e9
    print(f"\n  Enkoder            : {ENCODERS[args.encoder][0]} (zamrozony)")
    print(f"  Pacjentow          : {index.patient_id.nunique()}")
    print(f"  Kafelkow razem     : {N}")
    print(f"  Ksztalt na kafelek : ({n_tok}, {D})")
    print(f"  Rozmiar bufora     : {gb:.2f} GB (float16)")
    if gb > 8:
        print("  !! Duzy plik. Rozwaz mniejsze --patches-per-patient.")

    mode = "r+" if (args.resume and os.path.exists(emb_path)) else "w+"
    mm = np.lib.format.open_memmap(emb_path, mode=mode, dtype=np.float16,
                                   shape=(N, n_tok, D))

    start = 0
    if args.resume and os.path.exists(done_path):
        start = int(open(done_path).read().strip())
        print(f"  Wznowienie od kafelka {start}")

    paths = index["path"].tolist()
    bs = args.batch_size
    n_fail = 0

    print("\n  Przetwarzanie...")
    for i in range(start, N, bs):
        chunk = paths[i:i + bs]
        batch = np.zeros((len(chunk), args.img_size, args.img_size, 3), np.float32)
        for k, p in enumerate(chunk):
            try:
                im = Image.open(p).convert("RGB").resize((args.img_size, args.img_size))
                batch[k] = np.asarray(im, np.float32)
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    !! nie wczytano {p}: {e}")
        # preprocess ZAWSZE ta sama funkcja co przy inferencji - to bylo
        # zrodlo niezgodnosci w poprzedniej wersji pipeline'u
        mm[i:i + len(chunk)] = encoder.predict(preprocess(batch), verbose=0).astype(np.float16)

        if (i // bs) % 20 == 0:
            pct = min(i + bs, N) / N * 100
            print(f"    {min(i+bs,N):6d}/{N}  ({pct:5.1f}%)", flush=True)
            open(done_path, "w").write(str(i + len(chunk)))

    mm.flush()
    if os.path.exists(done_path):
        os.remove(done_path)

    print(f"\n  Nieudanych odczytow: {n_fail}")
    print("\n" + "=" * 74)
    print("ZAPISANO")
    print("=" * 74)
    print(f"  {emb_path}")
    print(f"  {idx_path}")
    print("\n  Nastepny krok:")
    print("    python train_eval_cv.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
