#!/usr/bin/env python3
"""
make_figures.py - ryciny 3-6 oraz tabela uzupelniajaca

Wejscie (pliki z katalogu cv_results/):
    cv_folds_er_final.csv       przebieg glowny, 3 ziarna  (wymagany)
    cv_folds_er_bycenter.csv    podzial po osrodku         (opcjonalny, Ryc. 6)

Wyjscie (katalog figures/):
    fig3_auc_boxplot.{png,pdf}
    fig4_rely_img.{png,pdf}
    fig5_variance.{png,pdf}
    fig6_replication.{png,pdf}
    table_S1_all_models.csv     pelne wyniki 450 modeli
    table_S2_summary.csv        statystyki zbiorcze
    table_S3_pairwise.csv       porownania par z korekta Holma

Uzycie:
    python make_figures.py
    python make_figures.py --results cv_results --out figures --dpi 600
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Kolejnosc ramion stala we wszystkich rycinach - ulatwia porownywanie
ARM_ORDER = ["tab", "img", "concat", "concat_wide", "cross", "cross_a2b",
             "cross_nores", "cross_a2b_nores"]
ARM_LABEL_PL = {
    "tab": "tylko kliniczne", "img": "tylko obraz",
    "concat": "konkatenacja", "concat_wide": "konkatenacja\n(szeroka)",
    "cross": "uwaga skro\u015bna\n(dwukierunkowa)", "cross_a2b": "uwaga skro\u015bna\n(tylko A\u2192B)",
    "cross_nores": "uwaga skro\u015bna\n(bez skr\u00f3tu)",
    "cross_a2b_nores": "uwaga skro\u015bna\n(A\u2192B, bez skr\u00f3tu)",
}
ARM_LABEL_EN = {
    "tab": "clinical only", "img": "image only",
    "concat": "concatenation", "concat_wide": "concatenation\n(wide)",
    "cross": "cross-attention\n(bidirectional)", "cross_a2b": "cross-attention\n(A\u2192B only)",
    "cross_nores": "cross-attention\n(no shortcut)",
    "cross_a2b_nores": "cross-attention\n(A\u2192B, no shortcut)",
}
TXT = {
 "pl": {
  "f3_title": "Skuteczno\u015b\u0107 klasyfikacji \u2014 \u017cadne por\u00f3wnanie nie przetrwa\u0142o korekty Holma",
  "auc_y": "AUC (poziom pacjenta)", "chance": "losowo", "ns": "n.i.",
  "f4_title": "Faktyczne wykorzystanie obrazu (spadek AUC po przetasowaniu wej\u015bcia)",
  "f4_x": "rely_img  =  AUC \u2212 AUC(obraz przetasowany)",
  "v_src": ["podzia\u0142\ndanych", "inicjalizacja\nwag", "architektura"],
  "v_y1": "odchylenie standardowe AUC", "v_y2": "udzia\u0142 w wariancji ca\u0142kowitej [%]",
  "v_t1": "Rozrzut wg \u017ar\u00f3d\u0142a", "v_t2": "Udzia\u0142 w wariancji",
  "v_sup": "Wariancja inicjalizacji przewy\u017csza architektoniczn\u0105 {r:.1f}-krotnie",
  "f6_title": "Stabilno\u015b\u0107 efektu w r\u00f3\u017cnych konfiguracjach",
  "f6_x": "\u0394 rely_img  (konkatenacja \u2212 uwaga skro\u015bna)",
  "f6_a": "3 ziarna\npodzia\u0142 losowy", "f6_b": "1 ziarno\npodzia\u0142 po o\u015brodku",
 },
 "en": {
  "f3_title": "Classification performance \u2014 no comparison survived Holm correction",
  "auc_y": "AUC (patient level)", "chance": "chance", "ns": "n.s.",
  "f4_title": "Actual use of the image (AUC drop after input permutation)",
  "f4_x": "rely_img  =  AUC \u2212 AUC(image permuted)",
  "v_src": ["data\nsplit", "weight\ninitialisation", "architecture"],
  "v_y1": "AUC standard deviation", "v_y2": "share of total variance [%]",
  "v_t1": "Dispersion by source", "v_t2": "Share of variance",
  "v_sup": "Initialisation variance exceeds architectural variance {r:.1f}-fold",
  "f6_title": "Effect stability across configurations",
  "f6_x": "\u0394 rely_img  (concatenation \u2212 cross-attention)",
  "f6_a": "3 seeds\nrandom split", "f6_b": "1 seed\nsite split",
 },
}
ARM_LABEL = ARM_LABEL_PL
T = TXT["pl"]

MULTIMODAL = ["concat", "concat_wide", "cross", "cross_a2b",
              "cross_nores", "cross_a2b_nores"]

# Paleta bezpieczna dla daltonistow (Okabe-Ito)
C_UNI, C_CONCAT, C_CROSS, C_GREY = "#0072B2", "#009E73", "#D55E00", "#666666"


def arm_color(a):
    if a in ("tab", "img"):
        return C_UNI
    if a.startswith("concat"):
        return C_CONCAT
    return C_CROSS


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save(fig, out, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {name}.png / .pdf")


def load(path):
    d = pd.read_csv(path)
    keys = ["rep", "fold", "arm"] + (["seed"] if "seed" in d.columns else [])
    return d.drop_duplicates(keys)


def per_fold(d, col="auc"):
    """Usrednia po ziarnach -> jedna wartosc na (rep, fold, arm)."""
    return d.groupby(["rep", "fold", "arm"])[col].mean().reset_index()


def holm(pvals):
    """Korekta Holma-Bonferroniego."""
    order = np.argsort(pvals)
    m = len(pvals)
    out = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, pvals[i] * (m - rank))
        prev = max(prev, val)          # wymuszenie monotonicznosci
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# Rycina 3 - AUC
# ---------------------------------------------------------------------------

def fig3(d, out, dpi):
    agg = per_fold(d, "auc")
    arms = [a for a in ARM_ORDER if a in agg.arm.unique()]
    data = [agg[agg.arm == a].auc.values for a in arms]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.3))
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(arm_color(a)); patch.set_alpha(0.28)
        patch.set_edgecolor(arm_color(a)); patch.set_linewidth(1.2)

    rng = np.random.RandomState(0)
    for i, (a, v) in enumerate(zip(arms, data), start=1):
        ax.scatter(rng.normal(i, 0.055, len(v)), v, s=11, alpha=0.55,
                   color=arm_color(a), zorder=3, linewidths=0)
        ax.scatter([i], [v.mean()], marker="D", s=26, color="black", zorder=4)

    ax.axhline(0.5, color=C_GREY, ls=":", lw=1)
    ax.text(len(arms) + 0.45, 0.505, T["chance"], color=C_GREY, fontsize=7, va="bottom")
    ax.set_xticks(range(1, len(arms) + 1))
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms])
    ax.set_ylabel(T["auc_y"])
    ax.set_title(T["f3_title"], loc="left")
    ax.set_ylim(0.40, 1.0)
    save(fig, out, "fig3_auc_boxplot", dpi)


# ---------------------------------------------------------------------------
# Rycina 4 - poleganie na obrazie
# ---------------------------------------------------------------------------

def fig4(d, out, dpi):
    agg = per_fold(d, "rely_img")
    arms = [a for a in MULTIMODAL if a in agg.arm.unique()]

    means, los, his, ps = [], [], [], []
    for a in arms:
        v = agg[agg.arm == a].rely_img.dropna().values
        se = v.std(ddof=1) / np.sqrt(len(v))
        means.append(v.mean()); los.append(1.96 * se); his.append(1.96 * se)
        ps.append(wilcoxon(v, alternative="greater")[1])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(arms))
    ax.errorbar(means, y, xerr=[los, his], fmt="o", ms=7, capsize=4,
                lw=1.6, color="black", zorder=3)
    for i, a in enumerate(arms):
        ax.scatter([means[i]], [y[i]], s=70, color=arm_color(a), zorder=4)
        star = "***" if ps[i] < 1e-3 else "**" if ps[i] < 1e-2 else \
               "*" if ps[i] < 0.05 else T["ns"]
        ax.text(means[i] + his[i] + 0.0018, y[i], star, va="center", fontsize=8)

    ax.axvline(0, color=C_GREY, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABEL.get(a, a).replace("\n", " ") for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel(T["f4_x"])
    ax.set_title(T["f4_title"], loc="left")
    save(fig, out, "fig4_rely_img", dpi)


# ---------------------------------------------------------------------------
# Rycina 5 - dekompozycja wariancji
# ---------------------------------------------------------------------------

def fig5(d, out, dpi):
    has_seed = "seed" in d.columns and d.seed.nunique() > 1
    sd_seed = (d.groupby(["rep", "fold", "arm"]).auc.std(ddof=1).mean()
               if has_seed else np.nan)
    agg = per_fold(d, "auc")
    P = agg.pivot_table(index=["rep", "fold"], columns="arm", values="auc")
    sd_fold, sd_arm = P.mean(axis=1).std(ddof=1), P.mean(axis=0).std(ddof=1)

    names = list(T["v_src"])
    sds = [sd_fold, sd_seed, sd_arm]
    cols = ["#7B3294", "#C2A5CF", C_GREY]
    keep = [i for i, v in enumerate(sds) if np.isfinite(v)]
    names = [names[i] for i in keep]; sds = [sds[i] for i in keep]
    cols = [cols[i] for i in keep]
    var = np.array(sds) ** 2
    share = var / var.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(names, sds, color=cols, alpha=0.85, width=0.6)
    for i, v in enumerate(sds):
        axes[0].text(i, v + 0.0018, f"{v:.4f}", ha="center", fontsize=8)
    axes[0].set_ylabel(T["v_y1"])
    axes[0].set_title(T["v_t1"], loc="left")

    axes[1].bar(names, share, color=cols, alpha=0.85, width=0.6)
    for i, v in enumerate(share):
        axes[1].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8)
    axes[1].set_ylabel(T["v_y2"])
    axes[1].set_ylim(0, 100)
    axes[1].set_title(T["v_t2"], loc="left")

    if len(sds) == 3:
        fig.suptitle(T["v_sup"].format(r=(sds[1]/sds[2])**2),
                     fontsize=10, x=0.01, ha="left")
    save(fig, out, "fig5_variance", dpi)


# ---------------------------------------------------------------------------
# Rycina 6 - niestabilnosc replikacji
# ---------------------------------------------------------------------------

def fig6(final, bycenter, out, dpi):
    """concat vs cross (rely_img) w roznych konfiguracjach."""
    rows = []

    def diff(d, label):
        a = per_fold(d, "rely_img")
        P = a.pivot_table(index=["rep", "fold"], columns="arm", values="rely_img")
        if not {"concat", "cross"} <= set(P.columns):
            return
        P = P.dropna(subset=["concat", "cross"])
        dv = (P["concat"] - P["cross"]).values
        se = dv.std(ddof=1) / np.sqrt(len(dv))
        rows.append((label, dv.mean(), 1.96 * se, wilcoxon(P["concat"], P["cross"])[1]))

    diff(final, T["f6_a"])
    if bycenter is not None:
        diff(bycenter, T["f6_b"])
    if not rows:
        print("  Ryc. 6 pominieta (brak ramion concat/cross)")
        return

    fig, ax = plt.subplots(figsize=(6.2, 2.4 + 0.5 * len(rows)))
    y = np.arange(len(rows))
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows],
                fmt="o", ms=7, capsize=4, lw=1.6, color="black")
    for i, r in enumerate(rows):
        ax.text(r[1] + r[2] + 0.0015, y[i], f"p={r[3]:.4f}", va="center", fontsize=8)
    ax.axvline(0, color=C_GREY, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel(T["f6_x"])
    ax.set_title(T["f6_title"], loc="left")
    save(fig, out, "fig6_replication", dpi)


# ---------------------------------------------------------------------------
# Tabele uzupelniajace
# ---------------------------------------------------------------------------

def tables(d, out):
    cols = [c for c in ["rep", "fold", "seed", "arm", "n_test", "auc", "ap",
                        "acc", "f1", "rely_img", "rely_tab", "auc_img_shuf",
                        "auc_tab_shuf", "best_epoch", "epochs_run"] if c in d.columns]
    s1 = d[cols].sort_values([c for c in ["rep", "fold", "arm", "seed"] if c in cols])
    s1.to_csv(os.path.join(out, "table_S1_all_models.csv"), index=False)
    print(f"  table_S1_all_models.csv  ({len(s1)} modeli)")

    met = [c for c in ["auc", "ap", "acc", "f1", "rely_img"] if c in d.columns]
    recs = []
    for a in [x for x in ARM_ORDER if x in d.arm.unique()]:
        r = {"arm": a}
        for m in met:
            v = per_fold(d, m).query("arm == @a")[m].dropna().values
            if len(v) == 0:
                continue
            se = v.std(ddof=1) / np.sqrt(len(v))
            r[f"{m}_mean"] = round(v.mean(), 4)
            r[f"{m}_sd"] = round(v.std(ddof=1), 4)
            r[f"{m}_ci_lo"] = round(v.mean() - 1.96 * se, 4)
            r[f"{m}_ci_hi"] = round(v.mean() + 1.96 * se, 4)
        recs.append(r)
    pd.DataFrame(recs).to_csv(os.path.join(out, "table_S2_summary.csv"), index=False)
    print("  table_S2_summary.csv")

    out_rows = []
    for metric in [m for m in ["auc", "rely_img"] if m in d.columns]:
        P = per_fold(d, metric).pivot_table(index=["rep", "fold"],
                                            columns="arm", values=metric)
        arms = [a for a in ARM_ORDER if a in P.columns and P[a].notna().any()]
        recs, ps = [], []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                sub = P[[a, b]].dropna()
                if len(sub) < 5:
                    continue
                _, p = wilcoxon(sub[a], sub[b])
                recs.append(dict(metric=metric, arm_a=a, arm_b=b,
                                 delta=round(sub[a].mean() - sub[b].mean(), 4),
                                 n_folds=len(sub), p_raw=p))
                ps.append(p)
        if recs:
            for r, ph in zip(recs, holm(np.array(ps))):
                r["p_holm"] = round(ph, 5)
                r["p_raw"] = round(r["p_raw"], 5)
                r["signif"] = "*" if ph < 0.05 else ""
            out_rows += recs
    pd.DataFrame(out_rows).sort_values(["metric", "p_raw"]).to_csv(
        os.path.join(out, "table_S3_pairwise.csv"), index=False)
    print("  table_S3_pairwise.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="cv_results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--lang", default="pl", choices=["pl", "en"])
    args = ap.parse_args()
    global ARM_LABEL, T
    ARM_LABEL = ARM_LABEL_EN if args.lang == "en" else ARM_LABEL_PL
    T = TXT[args.lang]

    os.makedirs(args.out, exist_ok=True)
    style()

    fp = os.path.join(args.results, "cv_folds_er_final.csv")
    if not os.path.exists(fp):
        raise SystemExit(f"Brak {fp}")
    final = load(fp)
    print(f"Wczytano {len(final)} modeli, ramiona: {sorted(final.arm.unique())}")

    bp = os.path.join(args.results, "cv_folds_er_bycenter.csv")
    bycenter = load(bp) if os.path.exists(bp) else None

    print("\nRyciny:")
    fig3(final, args.out, args.dpi)
    fig4(final, args.out, args.dpi)
    fig5(final, args.out, args.dpi)
    fig6(final, bycenter, args.out, args.dpi)

    print("\nTabele:")
    tables(final, args.out)
    print(f"\nGotowe: {args.out}/")


if __name__ == "__main__":
    main()
