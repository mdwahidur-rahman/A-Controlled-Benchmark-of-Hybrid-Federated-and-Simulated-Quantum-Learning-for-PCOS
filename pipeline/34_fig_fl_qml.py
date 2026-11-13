#!/usr/bin/env python3
"""Fig. 3 of the manuscript: (a) the federated track and (b) the simulated quantum slot, drawn as a vector schematic.
Only design facts that appear in 15_fl.py and 11_quantum.py are printed (K = 10 clients, Poisson group sampling, clipping, Gaussian noise, RDP accountant;
compressor -> tanh -> RY(pi u) -> L layers of Rot + CNOT -> <Z_i> -> linear read-out). No result is drawn.
Reuses the helpers of 31_schematics.py (measured text wrapping and a fit check).
Usage: PCOS_BENCH_ROOT=<repo root> OUT=<figure dir> PREV=<preview dir> python3 34_fig_fl_qml.py"""
import os, importlib
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
S = importlib.import_module("31_schematics")
box, arrow, line, put, canvas, PROBLEMS = S.box, S.arrow, S.line, S.put, S.canvas, S.PROBLEMS
INK, INK2, EDGE, FILL, ORANGE, PURPLE, TINT = S.INK, S.INK2, S.EDGE, S.FILL, S.ORANGE, S.PURPLE, S.TINT

def fig3():
    W, H = 7.0, 5.00; fig, ax = canvas(W, H)
    # ------------------------------------------------------------------ (a) federated track
    ax.text(0.04, H - 0.10, "(a) Federated track", fontsize=8.2, fontweight="bold", ha="left", va="center", color=INK)
    ax.text(1.50, H - 0.10, "frozen-encoder features; linear logistic head; duplicate groups as clients", fontsize=6.6, fontstyle="italic", ha="left", va="center", color=INK2)
    y0, h = 3.20, 1.30; yt = y0 + h
    ax.add_patch(FancyBboxPatch((0.04, 2.78), W - 0.08, H - 2.78 - 0.22, boxstyle="round,pad=0,rounding_size=0.04", fc=TINT[ORANGE], ec=ORANGE, lw=1.1, zorder=1))
    b1 = box(ax, 0.14, y0, 1.38, h, "Clients", "K = 10 clients receive whole duplicate groups (never a group split across clients); label, quantity, feature or combined skew", fc="white", ec=ORANGE, lw=0.8, fs=6.2, tfs=6.9, pad=0.05)
    b2 = box(ax, 1.84, y0, 2.78, h, None, None, fc="white", ec=ORANGE, lw=0.8, fs=6.2, tfs=6.9, pad=0.05)
    ax.text(1.84 + 1.39, yt - 0.12, "Local update on client k (E steps)", fontsize=6.9, fontweight="bold", ha="center", va="center", color=INK, zorder=5)
    items = [("FedAvg", "step on the group-weighted logistic loss with L2 penalty"), ("FedProx", "adds the proximal gradient μ(w − wₜ)"),
             ("SCAFFOLD", "adds the control-variate correction c − cₖ"), ("Group-level DP", "per-group gradient clipped at C, Gaussian noise σC, Poisson group sampling at rate q")]
    yy = yt - 0.34
    for t, b in items:
        ax.text(1.84 + 0.10, yy, t, fontsize=6.3, fontweight="bold", ha="left", va="center", color=INK)
        ls, hb = put(ax, 1.84 + 0.10 + 0.78, yy - 0.0, b, 6.1, ha="left", va="center", width=2.78 - 0.10 - 0.78 - 0.06, what=t)
        yy -= max(0.19, hb + 0.075)
    b3 = box(ax, 4.94, y0, 1.18, h, "Server", "weighted average of client models; weight proportional to the number of groups", fc="white", ec=ORANGE, lw=0.8, fs=6.2, tfs=6.9, pad=0.05)
    b4 = box(ax, 6.32, y0, 0.58, h, None, "Global head", fc="white", ec=ORANGE, lw=0.8, fs=6.4, pad=0.04, bold_body=True)
    arrow(ax, (0.14 + 1.38 + 0.01, y0 + h / 2), (1.84 - 0.01, y0 + h / 2), color=ORANGE, lw=1.0)
    arrow(ax, (1.84 + 2.78 + 0.01, y0 + h / 2), (4.94 - 0.01, y0 + h / 2), color=ORANGE, lw=1.0)
    arrow(ax, (4.94 + 1.18 + 0.01, y0 + h / 2), (6.32 - 0.01, y0 + h / 2), color=ORANGE, lw=1.0)
    ax.text(4.78, y0 + h / 2 + 0.10, "Δwₖ", fontsize=6.4, ha="center", va="center", color=INK2)
    # broadcast loop above the boxes
    yl = yt + 0.085
    line(ax, [(5.53, yt + 0.003), (5.53, yl), (3.23, yl)], color=ORANGE, lw=1.0); arrow(ax, (3.23, yl), (3.23, yt + 0.004), color=ORANGE, lw=1.0)
    ax.text(4.38, yl + 0.065, "broadcast wₜ₊₁, repeat for R rounds", fontsize=6.2, ha="center", va="center", color=INK2, bbox=dict(fc=TINT[ORANGE], ec="none", pad=0.4))
    # accountant and selection strip
    box(ax, 1.84, 2.87, 2.78, 0.27, None, "RDP accountant: (q, σ, steps) → (ε, δ = 10⁻⁵)", fc="white", ec=ORANGE, lw=0.7, fs=6.3, pad=0.04)
    box(ax, 4.94, 2.87, 1.96, 0.27, None, "rounds and settings chosen on validation AUC only", fc="white", ec=ORANGE, lw=0.7, fs=6.2, pad=0.04)
    arrow(ax, (3.23, y0 - 0.003), (3.23, 3.145), color=ORANGE, lw=0.8, ls=(0, (3, 2)))
    # ------------------------------------------------------------------ (b) quantum slot
    pb = 0.03; ph = 2.44
    ax.add_patch(FancyBboxPatch((0.04, pb), W - 0.08, ph - 0.00, boxstyle="round,pad=0,rounding_size=0.04", fc=TINT[PURPLE], ec=PURPLE, lw=1.1, zorder=1))
    ax.text(0.12, pb + ph - 0.13, "(b) Simulated quantum slot", fontsize=8.2, fontweight="bold", ha="left", va="center", color=INK)
    ax.text(1.95, pb + ph - 0.13, "noiseless statevector simulation; no quantum hardware", fontsize=6.6, fontstyle="italic", ha="left", va="center", color=INK2)
    by, bh = 0.62, 1.40
    c1 = box(ax, 0.14, by, 0.92, bh, "Feature", "pooled feature, standardized with training statistics", fc="white", ec=PURPLE, lw=0.8, fs=6.2, tfs=6.9, pad=0.04)
    c2 = box(ax, 1.30, by, 1.10, bh, "Compressor", "linear D → n, then tanh: uᵢ in (−1, 1)", fc="white", ec=PURPLE, lw=0.8, fs=6.2, tfs=6.9, pad=0.04)
    arrow(ax, (0.14 + 0.92 + 0.01, by + bh / 2), (1.30 - 0.01, by + bh / 2), color=PURPLE, lw=1.0)
    # circuit panel
    cx0, cx1 = 2.64, 5.52; ax.add_patch(FancyBboxPatch((cx0, by), cx1 - cx0, bh, boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=PURPLE, lw=0.8, zorder=2))
    arrow(ax, (1.30 + 1.10 + 0.01, by + bh / 2), (cx0 - 0.01, by + bh / 2), color=PURPLE, lw=1.0)
    ax.text((cx0 + cx1) / 2, by + bh - 0.095, "Variational circuit (n = 4 qubits, L = 2 layers drawn)", fontsize=6.5, fontweight="bold", ha="center", va="center", color=INK, zorder=5)
    qy = [by + 1.00 - 0.20 * k for k in range(4)]; lx0, lx1 = cx0 + 0.30, cx1 - 0.06
    for k, y in enumerate(qy):
        ax.text(cx0 + 0.05, y, f"q{k + 1}", fontsize=6.0, ha="left", va="center", color=INK2, zorder=5); ax.plot([lx0, lx1], [y, y], color=INK2, lw=0.7, zorder=3)
    def gate(x, y, txt, w=0.30, hh=0.15, fc="#eeeaf8", ec=PURPLE):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - hh / 2), w, hh, boxstyle="round,pad=0,rounding_size=0.015", fc=fc, ec=ec, lw=0.8, zorder=4)); ax.text(x, y, txt, fontsize=5.5, ha="center", va="center", color=INK, zorder=5)
    def cnot(x, yc, yt_):
        ax.plot([x, x], [yc, yt_], color=INK, lw=0.8, zorder=4); ax.add_patch(Circle((x, yc), 0.020, fc=INK, ec=INK, zorder=5))
        ax.add_patch(Circle((x, yt_), 0.045, fc="white", ec=INK, lw=0.8, zorder=5)); ax.plot([x - 0.045, x + 0.045], [yt_, yt_], color=INK, lw=0.8, zorder=6); ax.plot([x, x], [yt_ - 0.045, yt_ + 0.045], color=INK, lw=0.8, zorder=6)
    xe, xr1, xc1, xr2, xc2, xm = cx0 + 0.52, cx0 + 1.02, cx0 + 1.40, cx0 + 1.80, cx0 + 2.18, cx0 + 2.56
    for y in qy: gate(xe, y, "RY(πu)", w=0.34, fc="#f4f1fb")
    for y in qy: gate(xr1, y, "Rot"); gate(xr2, y, "Rot")
    cnot(xc1, qy[0], qy[1]); cnot(xc1, qy[2], qy[3]); cnot(xc2, qy[1], qy[2])
    for y in qy:
        ax.add_patch(FancyBboxPatch((xm - 0.14, y - 0.075), 0.28, 0.15, boxstyle="round,pad=0,rounding_size=0.015", fc="#fff7e0", ec=EDGE, lw=0.8, zorder=4)); ax.text(xm, y, "⟨Z⟩", fontsize=5.6, ha="center", va="center", color=INK, zorder=5)
    # layer bracket
    yb = qy[-1] - 0.145; line(ax, [(xr1 - 0.19, yb), (xr1 - 0.19, yb - 0.03), (xc2 + 0.10, yb - 0.03), (xc2 + 0.10, yb)], color=INK2, lw=0.7, z=5)
    ax.text((xr1 - 0.19 + xc2 + 0.10) / 2, yb - 0.095, "repeated for L layers", fontsize=5.8, ha="center", va="center", color=INK2, zorder=5)
    ax.text(xe, yb - 0.095, "encoding", fontsize=5.8, ha="center", va="center", color=INK2, zorder=5)
    ax.text(xm, yb - 0.095, "read-out", fontsize=5.8, ha="center", va="center", color=INK2, zorder=5)
    # read-out box
    c4 = box(ax, 5.78, by, 1.12, bh, "Read-out", "Linear n → 1 on ⟨Zᵢ⟩, then sigmoid: p(PCOS-positive)", fc="white", ec=PURPLE, lw=0.8, fs=6.2, tfs=6.9, pad=0.04)
    arrow(ax, (cx1 + 0.01, by + bh / 2), (5.78 - 0.01, by + bh / 2), color=PURPLE, lw=1.0)
    # notes strip
    note = ("Rot = RZ·RY·RZ with three trainable angles per qubit and layer (3nL slot parameters; 54 at n = 6, L = 3). CNOT pattern: brick (nearest neighbors) or ring. "
            "Setting B uses the slot as a hybrid layer; setting A adds a classical linear branch on the same feature. Controls: identity, dense layers, deranged partner.")
    put(ax, W / 2, pb + 0.28, note, 6.1, "normal", INK2, width=W - 0.40, what="note b")
    fig.savefig(f"{S.OUT}/fig03_fl_qml.pdf"); fig.savefig(f"{S.PREV}/fig03_fl_qml.png", dpi=230); plt.close(fig)

if __name__ == "__main__":
    fig3(); print("fit problems:", len(PROBLEMS)); [print("  ", p) for p in PROBLEMS]
