#!/usr/bin/env python3
"""
make_fig1_fig2.py — ryciny koncepcyjne 1 i 2

Rycina 1: schemat architektury dwukierunkowej uwagi skrośnej z zaznaczeniem
          ścieżek rezydualnych, którymi faktycznie płynie sygnał.
Rycina 2: ilustracja degeneracji kierunku B→A — dwa różne obrazy,
          identyczne wyjście modułu uwagi.

Użycie:
    python make_fig1_fig2.py --out figures --dpi 600
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_TAB, C_IMG, C_ATT, C_RES, C_GREY = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#666666"

# Etykiety w dwoch jezykach - przelacznik --lang
LANG = {
 "pl": {
  "clin": "dane kliniczne\n(26 cech)", "img": "obraz H&E\n(49 tokenów × 1280)",
  "a2b": "kierunek A\u2192B\nuwaga(q, kv, kv)\n\u00ab klinika pyta obraz \u00bb",
  "b2a": "kierunek B\u2192A\nuwaga(kv, q, q)\n\u00ab obraz pyta klinik\u0119 \u00bb",
  "res_q": "po\u0142\u0105czenie rezydualne zapytania \u2014 bezpo\u015brednia \u015bcie\u017cka kliniczna",
  "res_kv": "po\u0142\u0105czenie rezydualne token\u00f3w obrazowych \u2014 G\u0141\u00d3WNA \u015bcie\u017cka sygna\u0142u obrazowego",
  "degen": "softmax po 1 elemencie \u2261 1  \u2192  wyj\u015bcie = wektor kliniczny,\n"
           "niezale\u017cnie od obrazu (kierunek zdegenerowany)",
  "leg_res": "\u015bcie\u017cki rezydualne (omijaj\u0105 uwag\u0119)", "leg_att": "modu\u0142y uwagi",
  "head": "konkat.\n\u2192 g\u0142owa\n\u2192 p(HR+)",
  "f2_title": "Degeneracja kierunku B\u2192A: przy jednoelementowym kluczu wyj\u015bcie uwagi\n"
              "nie zale\u017cy od obrazu i jest sta\u0142e wzd\u0142u\u017c wszystkich 49 pozycji  "
              "(maks. |\u0394| = 0,0 \u2014 weryfikacja numeryczna)",
  "f2_img1": "obraz nr 1\n(tokeny 7\u00d77, projekcja kv)",
  "f2_img2": "obraz nr 2 \u2014 INNY\n(tokeny 7\u00d77, projekcja kv)",
  "f2_box": "uwaga B\u2192A\nsoftmax po jednym\nelemencie \u2261 1",
  "f2_vec": "ten sam wektor kliniczny (1 token)",
  "f2_out1": "wyj\u015bcie dla obrazu nr 1",
  "f2_out2": "wyj\u015bcie dla obrazu nr 2\nIDENTYCZNE co do bitu",
 },
 "en": {
  "clin": "clinical data\n(26 features)", "img": "H&E image\n(49 tokens \u00d7 1280)",
  "a2b": "direction A\u2192B\nattention(q, kv, kv)\n\u00ab clinical queries image \u00bb",
  "b2a": "direction B\u2192A\nattention(kv, q, q)\n\u00ab image queries clinical \u00bb",
  "res_q": "query residual connection \u2014 direct clinical pathway",
  "res_kv": "image-token residual connection \u2014 MAIN image-signal pathway",
  "degen": "softmax over 1 element \u2261 1  \u2192  output = clinical vector,\n"
           "independent of the image (degenerate direction)",
  "leg_res": "residual pathways (bypass attention)", "leg_att": "attention modules",
  "head": "concat.\n\u2192 head\n\u2192 p(HR+)",
  "f2_title": "Degeneracy of the B\u2192A direction: with a single-element key the attention\n"
              "output does not depend on the image and is constant across all 49 positions  "
              "(max |\u0394| = 0.0 \u2014 numerical verification)",
  "f2_img1": "image 1\n(7\u00d77 tokens, kv projection)",
  "f2_img2": "image 2 \u2014 DIFFERENT\n(7\u00d77 tokens, kv projection)",
  "f2_box": "B\u2192A attention\nsoftmax over one\nelement \u2261 1",
  "f2_vec": "same clinical vector (1 token)",
  "f2_out1": "output for image 1",
  "f2_out2": "output for image 2\nBITWISE IDENTICAL",
 },
}
L = LANG["pl"]     # ustawiane w main()


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10,
        "figure.constrained_layout.use": False})


def save(fig, out, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  zapisano {name}.png / .pdf")


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="black", lw=1.2, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=lw, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.4, style="-", curve=0.0, z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f"arc3,rad={curve}", arrowstyle="-|>",
                 mutation_scale=11, color=color, lw=lw, linestyle=style, zorder=z))


# ---------------------------------------------------------------------------
# Rycina 1 — schemat architektury
# ---------------------------------------------------------------------------

def fig1(out, dpi):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    # wejścia
    box(ax, 2, 44, 20, 7, L["clin"], fc="#E3F0FA", ec=C_TAB, lw=1.6)
    box(ax, 2, 11, 20, 7, L["img"], fc="#E4F5EF", ec=C_IMG, lw=1.6)

    # projekcje
    box(ax, 27, 44, 13, 7, "Dense\n→ q (1×128)", ec=C_TAB)
    box(ax, 27, 11, 13, 7, "Dense\n→ kv (49×128)", ec=C_IMG)
    arrow(ax, 22, 47.5, 27, 47.5, C_TAB)
    arrow(ax, 22, 14.5, 27, 14.5, C_IMG)

    # ===== kierunek A->B =====
    box(ax, 47, 42, 22, 11,
        L["a2b"], ec=C_ATT, lw=1.6)
    arrow(ax, 40, 47.5, 47, 47.5, C_TAB)
    arrow(ax, 40, 14.5, 44, 14.5, C_IMG)
    arrow(ax, 44, 14.5, 47, 44.5, C_IMG, curve=-0.25)

    # skrot rezydualny q (nad blokiem)
    arrow(ax, 33.5, 51, 33.5, 57, C_RES, lw=2.2)
    arrow(ax, 33.5, 57, 74, 57, C_RES, lw=2.2)
    arrow(ax, 74, 57, 74, 50, C_RES, lw=2.2)
    ax.text(53, 58.3, L["res_q"],
            fontsize=8, color=C_RES, ha="center")

    # ===== kierunek B->A =====
    box(ax, 47, 8, 22, 11,
        L["b2a"], ec=C_ATT, lw=1.6)
    arrow(ax, 40, 14.5, 47, 13.5, C_IMG)
    arrow(ax, 40, 47.5, 44, 47.5, C_TAB)
    arrow(ax, 44, 47.5, 47, 16.5, C_TAB, curve=0.25)

    ax.text(58, 22.6, L["degen"], fontsize=8, color=C_ATT, ha="center", style="italic")

    # skrot rezydualny kv (pod blokiem)
    arrow(ax, 33.5, 11, 33.5, 2.5, C_RES, lw=2.6)
    arrow(ax, 33.5, 2.5, 74, 2.5, C_RES, lw=2.6)
    arrow(ax, 74, 2.5, 74, 12, C_RES, lw=2.6)
    ax.text(53, 0.4, L["res_kv"],
            fontsize=8, color=C_RES, ha="center", fontweight="bold")

    # poolingi i konkatenacja
    box(ax, 72.5, 42, 10, 7, "LN, FFN\nGAP", fs=8)
    box(ax, 72.5, 11, 10, 7, "LN, FFN\nGAP", fs=8)
    arrow(ax, 69, 47.5, 72.5, 45.5)
    arrow(ax, 69, 13.5, 72.5, 14.5)

    box(ax, 87, 26, 11, 9, L["head"], fc="#F2F2F2")
    arrow(ax, 82.5, 45.5, 87, 32, curve=-0.15)
    arrow(ax, 82.5, 14.5, 87, 29, curve=0.15)

    # legenda — pusty obszar po lewej, między wejściami
    ax.plot([3, 7], [33, 33], color=C_RES, lw=2.4)
    ax.text(8, 33, L["leg_res"], fontsize=8, va="center")
    ax.plot([3, 7], [29.5, 29.5], color=C_ATT, lw=1.6)
    ax.text(8, 29.5, L["leg_att"], fontsize=8, va="center")

    save(fig, out, "fig1_architektura", dpi)


# ---------------------------------------------------------------------------
# Rycina 2 — degeneracja B->A
# ---------------------------------------------------------------------------

def fig2(out, dpi):
    rng = np.random.RandomState(0)
    A = rng.rand(7, 7)
    B = rng.rand(7, 7)
    OUT = np.tile(np.random.RandomState(42).rand(1, 7), (7, 1))  # stałe po pozycjach

    fig = plt.figure(figsize=(8.6, 4.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.28, 1.05, 0.28, 1],
                          height_ratios=[1, 1], wspace=0.06, hspace=0.45)

    def grid(axp, data, title, cmap, tcolor="black"):
        axp.imshow(data, cmap=cmap, vmin=0, vmax=1)
        axp.set_xticks([]); axp.set_yticks([])
        axp.set_title(title, fontsize=8.5, color=tcolor)

    axA = fig.add_subplot(gs[0, 0])
    grid(axA, A, L["f2_img1"], "Greens")
    axB = fig.add_subplot(gs[1, 0])
    grid(axB, B, L["f2_img2"], "Greens")

    for row in (0, 1):
        axm = fig.add_subplot(gs[row, 2]); axm.axis("off")
        axm.set_xlim(0, 10); axm.set_ylim(0, 10)
        box(axm, 0.4, 3.2, 9.2, 4.2,
            L["f2_box"],
            ec=C_ATT, lw=1.5, fs=8)
        arrow(axm, -0.6, 5.3, 0.4, 5.3, "black")
        arrow(axm, 9.6, 5.3, 10.8, 5.3, "black")
        axm.text(5, 1.3, L["f2_vec"],
                 fontsize=7.5, ha="center", color=C_TAB)
        arrow(axm, 5, 0.2, 5, 3.2, C_TAB, lw=1.3)

    ax1 = fig.add_subplot(gs[0, 4])
    grid(ax1, OUT, L["f2_out1"], "Purples")
    ax2 = fig.add_subplot(gs[1, 4])
    grid(ax2, OUT, L["f2_out2"], "Purples", tcolor="#7B3294")

    fig.suptitle(L["f2_title"], fontsize=9.5, y=1.04)
    save(fig, out, "fig2_degeneracja", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--lang", default="pl", choices=["pl", "en"])
    a = ap.parse_args()
    global L
    L = LANG[a.lang]
    os.makedirs(a.out, exist_ok=True)
    style()
    fig1(a.out, a.dpi)
    fig2(a.out, a.dpi)


if __name__ == "__main__":
    main()
