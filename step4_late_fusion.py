#!/usr/bin/env python3
"""
step4_late_fusion.py — fuzja pozna i analiza zgodnosci modalnosci

Odpowiada na pytanie, ktorego fuzja wczesna nie rozwiazala: skoro model obrazowy
i kliniczny niosa czesciowo rozbiezne sygnaly, to czy zamiast wymuszac wspolna
reprezentacje nie lepiej potraktowac je jako WZAJEMNA KONTROLE?

Trzy analizy:

  1. FUZJA POZNA — usrednienie prawdopodobienstw modeli jednomodalnych
     (arytmetyczne i rangowe) porownane z fuzja wczesna. Fuzja pozna nie wymusza
     wspolnej reprezentacji, wiec nie usrednia rozbieznych sygnalow ze strata.

  2. ZGODNOSC MODALNOSCI — podzial pacjentow na podgrupe zgodna (oba modele
     wskazuja ten sam status) i niezgodna. Jesli dokladnosc w podgrupie zgodnej
     jest istotnie wyzsza, niezgodnosc stanowi sygnal do priorytetowej weryfikacji
     immunohistochemicznej.

  3. PREDYKCJA SELEKTYWNA — krzywa dokladnosc/pokrycie przy odrzucaniu przypadkow
     najbardziej niepewnych. Miara |p_obraz - p_klin| porownana z klasyczna
     pewnoscia pojedynczego modelu max(p, 1-p): sprawdzamy, czy niezgodnosc
     modalnosci wnosi cos PONAD to, co widac w samym modelu.

Wejscie:
    cv_results/cv_preds_er_final.csv   (step3 z flaga --save-preds)

Uzycie:
    python step4_late_fusion.py
    python step4_late_fusion.py --preds cv_results/cv_preds_er_final.csv --out figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

C_UNI, C_LATE, C_EARLY, C_GREY = "#0072B2", "#CC79A7", "#009E73", "#666666"

TXT = {
 "pl": {
  "f7_title": "Fuzja p\u00f3\u017ana nie wymusza wsp\u00f3lnej reprezentacji",
  "auc_x": "AUC (poziom pacjenta)",
  "conc": "zgodne", "disc": "niezgodne",
  "f8_y": "dok\u0142adno\u015b\u0107 (fuzja p\u00f3\u017ana)",
  "f8_t1": "Zgodno\u015b\u0107 modalno\u015bci   p={p:.4f}",
  "f8_x2": "odsetek przypadk\u00f3w zgodnych w foldzie", "f8_y2": "liczba fold\u00f3w",
  "f8_t2": "\u015arednio {f:.0f}% zgodnych",
  "f9_title": "Krzywa predykcji selektywnej",
  "f9_x": "pokrycie [%]  (odsetek przypadk\u00f3w, na kt\u00f3re model odpowiada)",
  "f9_y": "dok\u0142adno\u015b\u0107 w zaakceptowanej podgrupie",
  "f9_dis": "niezgodno\u015b\u0107 modalno\u015bci", "f9_conf": "pewno\u015b\u0107 modelu (odniesienie)",
 },
 "en": {
  "f7_title": "Late fusion does not impose a shared representation",
  "auc_x": "AUC (patient level)",
  "conc": "concordant", "disc": "discordant",
  "f8_y": "accuracy (late fusion)",
  "f8_t1": "Modality agreement   p={p:.4f}",
  "f8_x2": "fraction of concordant cases per fold", "f8_y2": "number of folds",
  "f8_t2": "Mean {f:.0f}% concordant",
  "f9_title": "Selective prediction curve",
  "f9_x": "coverage [%]  (fraction of cases the model answers)",
  "f9_y": "accuracy in the accepted subgroup",
  "f9_dis": "modality disagreement", "f9_conf": "model confidence (reference)",
 },
}
T = TXT["pl"]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True})


def save(fig, out, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {name}.png / .pdf")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m, out, prev = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, p[i] * (m - rank)))
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# Przygotowanie: jeden wiersz na (fold, ziarno, pacjent), kolumny = ramiona
# ---------------------------------------------------------------------------

def build_wide(preds):
    w = preds.pivot_table(index=["rep", "fold", "seed", "patient_id", "y_true"],
                          columns="arm", values="y_prob").reset_index()
    w.columns.name = None
    return w


def add_fusions(w):
    """Fuzje pozne liczone WYLACZNIE z modeli jednomodalnych."""
    if not {"tab", "img"} <= set(w.columns):
        raise SystemExit("Brak ramion 'tab' i 'img' — fuzja pozna niemozliwa.")

    w = w.dropna(subset=["tab", "img"]).copy()
    w["late_mean"] = (w["tab"] + w["img"]) / 2

    # Usrednienie rangowe: odporne na rozne kalibracje obu modeli.
    # Rangowanie percentylowe wewnatrz kazdego (rep, fold, seed) - wektorowo,
    # bez groupby.apply, ktore przy zwracaniu tablic gubi wyrownanie indeksu.
    g = w.groupby(["rep", "fold", "seed"])
    w["late_rank"] = (g["tab"].rank(pct=True) + g["img"].rank(pct=True)) / 2
    return w


def per_fold_metric(w, cols, fn):
    """Metryka liczona osobno w kazdym (rep, fold, seed), potem usredniona po ziarnach."""
    recs = []
    for (rep, fold, seed), g in w.groupby(["rep", "fold", "seed"]):
        if g.y_true.nunique() < 2:
            continue
        r = dict(rep=rep, fold=fold, seed=seed)
        for c in cols:
            if c in g.columns and g[c].notna().all():
                r[c] = fn(g.y_true.values, g[c].values)
        recs.append(r)
    d = pd.DataFrame(recs)
    return d.groupby(["rep", "fold"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index()


# ---------------------------------------------------------------------------
# 1. Fuzja pozna vs wczesna
# ---------------------------------------------------------------------------

def analyse_late(w, out, dpi):
    cand = [c for c in ["tab", "img", "concat", "concat_wide", "cross",
                        "cross_a2b", "late_mean", "late_rank"] if c in w.columns]
    auc = per_fold_metric(w, cand, roc_auc_score)
    ap = per_fold_metric(w, cand, average_precision_score)

    print("\n=== 1. FUZJA POZNA vs WCZESNA (AUC, 25 foldow) ===")
    rows = []
    for c in cand:
        if c not in auc.columns:
            continue
        v = auc[c].dropna().values
        se = v.std(ddof=1) / np.sqrt(len(v))
        rows.append(dict(model=c, auc=v.mean(), lo=v.mean() - 1.96 * se,
                         hi=v.mean() + 1.96 * se,
                         ap=ap[c].mean() if c in ap.columns else np.nan))
        print(f"  {c:12s} AUC={v.mean():.3f}  95%CI=[{v.mean()-1.96*se:.3f}, "
              f"{v.mean()+1.96*se:.3f}]")
    summ = pd.DataFrame(rows)

    ref = [c for c in ["img", "concat", "cross"] if c in auc.columns]
    tests, ps = [], []
    for base in ["late_mean", "late_rank"]:
        if base not in auc.columns:
            continue
        for r in ref:
            sub = auc[[base, r]].dropna()
            _, p = wilcoxon(sub[base], sub[r])
            tests.append(dict(a=base, b=r, delta=sub[base].mean() - sub[r].mean(), p=p))
            ps.append(p)
    if tests:
        for t, ph in zip(tests, holm(ps)):
            t["p_holm"] = ph
        print("\n  Porownania (Wilcoxon + Holm):")
        for t in sorted(tests, key=lambda x: x["p"]):
            print(f"    {t['a']:11s} vs {t['b']:11s} dAUC={t['delta']:+.3f}  "
                  f"p={t['p']:.4f}  p_Holm={t['p_holm']:.4f}"
                  f"  {'*' if t['p_holm'] < 0.05 else ''}")

    order = summ.sort_values("auc")
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.6))
    y = np.arange(len(order))
    cols = [C_LATE if m.startswith("late") else C_UNI if m in ("tab", "img")
            else C_EARLY for m in order.model]
    ax.errorbar(order.auc, y, xerr=[order.auc - order.lo, order.hi - order.auc],
                fmt="none", capsize=4, lw=1.5, color="black", zorder=2)
    ax.scatter(order.auc, y, s=70, c=cols, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(order.model)
    ax.axvline(0.5, color=C_GREY, ls=":", lw=1)
    ax.set_xlabel(T["auc_x"])
    ax.set_title(T["f7_title"], loc="left")
    save(fig, out, "fig7_late_fusion", dpi)
    return summ, pd.DataFrame(tests)


# ---------------------------------------------------------------------------
# 2. Zgodnosc modalnosci
# ---------------------------------------------------------------------------

def analyse_agreement(w, out, dpi, thr=0.5):
    w = w.copy()
    w["pred_tab"] = (w["tab"] > thr).astype(int)
    w["pred_img"] = (w["img"] > thr).astype(int)
    w["zgodne"] = (w.pred_tab == w.pred_img).astype(int)
    w["poprawne_img"] = (w.pred_img == w.y_true).astype(int)
    w["poprawne_late"] = ((w["late_mean"] > thr).astype(int) == w.y_true).astype(int)

    print("\n=== 2. ZGODNOSC MODALNOSCI ===")
    frac = w.zgodne.mean()
    print(f"  Odsetek przypadkow zgodnych: {frac*100:.1f}%")

    recs = []
    for (rep, fold), g in w.groupby(["rep", "fold"]):
        z, n = g[g.zgodne == 1], g[g.zgodne == 0]
        recs.append(dict(rep=rep, fold=fold, frac=g.zgodne.mean(),
                         acc_z=z.poprawne_late.mean() if len(z) else np.nan,
                         acc_n=n.poprawne_late.mean() if len(n) else np.nan,
                         n_z=len(z), n_n=len(n)))
    a = pd.DataFrame(recs)
    sub = a[["acc_z", "acc_n"]].dropna()
    _, p = wilcoxon(sub.acc_z, sub.acc_n)

    print(f"  Dokladnosc w podgrupie ZGODNEJ    : {sub.acc_z.mean():.3f}")
    print(f"  Dokladnosc w podgrupie NIEZGODNEJ : {sub.acc_n.mean():.3f}")
    print(f"  Roznica: {sub.acc_z.mean()-sub.acc_n.mean():+.3f}   p={p:.5f}"
          f"  {'*' if p < 0.05 else ''}")
    print(f"  Sredni rozmiar podgrupy niezgodnej: {a.n_n.mean():.1f} z "
          f"{(a.n_z+a.n_n).mean():.1f} pacjentow na fold")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    axes[0].bar([T["conc"], T["disc"]], [sub.acc_z.mean(), sub.acc_n.mean()],
                color=[C_LATE, C_GREY], alpha=0.85, width=0.55)
    for i, v in enumerate([sub.acc_z.mean(), sub.acc_n.mean()]):
        axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_ylabel(T["f8_y"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(T["f8_t1"].format(p=p), loc="left")

    axes[1].hist(a.frac, bins=12, color=C_LATE, alpha=0.8)
    axes[1].axvline(a.frac.mean(), color="black", ls="--", lw=1.2)
    axes[1].set_xlabel(T["f8_x2"])
    axes[1].set_ylabel(T["f8_y2"])
    axes[1].set_title(T["f8_t2"].format(f=frac*100), loc="left")
    save(fig, out, "fig8_agreement", dpi)
    return a, p


# ---------------------------------------------------------------------------
# 3. Predykcja selektywna
# ---------------------------------------------------------------------------

def analyse_selective(w, out, dpi, thr=0.5):
    """
    Dwie miary niepewnosci porownane uczciwie:
      - niezgodnosc modalnosci |p_img - p_tab|   (wymaga obu modeli)
      - pewnosc pojedynczego modelu 1-|2p-1|     (odniesienie)
    Jesli pierwsza nie bije drugiej, zgodnosc nie wnosi nic PONAD to,
    co widac juz w samym modelu.
    """
    w = w.copy()
    w["u_disagree"] = (w["img"] - w["tab"]).abs()
    w["u_conf"] = 1 - (2 * w["late_mean"] - 1).abs()
    w["ok"] = ((w["late_mean"] > thr).astype(int) == w.y_true).astype(int)

    covs = np.round(np.arange(0.3, 1.001, 0.05), 2)
    curves = {}
    for name in ["u_disagree", "u_conf"]:
        rows = []
        for (rep, fold), g in w.groupby(["rep", "fold"]):
            g = g.sort_values(name)                 # najpewniejsze najpierw
            for c in covs:
                k = max(1, int(round(len(g) * c)))
                rows.append(dict(rep=rep, fold=fold, cov=c,
                                 acc=g.ok.iloc[:k].mean()))
        curves[name] = pd.DataFrame(rows).groupby("cov").acc.agg(["mean", "sem"])

    print("\n=== 3. PREDYKCJA SELEKTYWNA ===")
    print(f"  {'pokrycie':>9} {'niezgodnosc':>13} {'pewnosc modelu':>16}")
    for c in [0.5, 0.7, 0.9, 1.0]:
        if c in curves["u_disagree"].index:
            print(f"  {c:>9.0%} {curves['u_disagree'].loc[c,'mean']:>13.3f} "
                  f"{curves['u_conf'].loc[c,'mean']:>16.3f}")

    at = w.groupby(["rep", "fold"], group_keys=False)
    def acc_at(name, c):
        r = []
        for _, g in w.groupby(["rep", "fold"]):
            g = g.sort_values(name)
            k = max(1, int(round(len(g) * c)))
            r.append(g.ok.iloc[:k].mean())
        return np.array(r)
    d5, c5 = acc_at("u_disagree", 0.5), acc_at("u_conf", 0.5)
    _, p50 = wilcoxon(d5, c5)
    print(f"\n  Przy pokryciu 50%: niezgodnosc={d5.mean():.3f} vs "
          f"pewnosc={c5.mean():.3f}  d={d5.mean()-c5.mean():+.3f}  p={p50:.4f}"
          f"  {'*' if p50 < 0.05 else ''}")
    if p50 >= 0.05:
        print("  -> niezgodnosc modalnosci NIE wnosi nic ponad pewnosc pojedynczego modelu")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for name, lab, col in [("u_disagree", T["f9_dis"], C_LATE),
                           ("u_conf", T["f9_conf"], C_GREY)]:
        cur = curves[name]
        ax.plot(cur.index * 100, cur["mean"], "-o", ms=4, color=col, label=lab)
        ax.fill_between(cur.index * 100, cur["mean"] - 1.96 * cur["sem"],
                        cur["mean"] + 1.96 * cur["sem"], color=col, alpha=0.15)
    ax.set_xlabel(T["f9_x"])
    ax.set_ylabel(T["f9_y"])
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(T["f9_title"], loc="left")
    save(fig, out, "fig9_selective", dpi)
    return curves, p50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cv_results/cv_preds_er_final.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lang", default="pl", choices=["pl", "en"])
    args = ap.parse_args()
    global T
    T = TXT[args.lang]

    if not os.path.exists(args.preds):
        raise SystemExit(f"Brak {args.preds}\n"
                         f"Uruchom step3 z flaga --save-preds.")
    os.makedirs(args.out, exist_ok=True)
    style()

    preds = pd.read_csv(args.preds).drop_duplicates(
        ["rep", "fold", "seed", "arm", "patient_id"])
    print(f"Wczytano {len(preds)} predykcji | ramiona: {sorted(preds.arm.unique())}"
          f" | pacjenci: {preds.patient_id.nunique()}")

    w = add_fusions(build_wide(preds))
    summ, tests = analyse_late(w, args.out, args.dpi)
    agree, p_agree = analyse_agreement(w, args.out, args.dpi, args.threshold)
    curves, p_sel = analyse_selective(w, args.out, args.dpi, args.threshold)

    summ.to_csv(os.path.join(args.out, "table_S4_late_fusion.csv"), index=False)
    if len(tests):
        tests.to_csv(os.path.join(args.out, "table_S5_late_vs_early.csv"), index=False)
    agree.to_csv(os.path.join(args.out, "table_S6_agreement.csv"), index=False)
    print(f"\nGotowe: {args.out}/")


if __name__ == "__main__":
    main()
