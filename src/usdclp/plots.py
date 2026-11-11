"""Graficos del informe (PNG estaticos para el repositorio)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})


def equity_with_exposure(curves: dict[str, pd.Series], pos: pd.Series, title: str, path, colors=None):
    colors = colors or [BLUE, ORANGE, AQUA, MUTED, VIOLET]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.axhline(100, color=GRID, lw=0.8, zorder=0)
    for (name, s), c in zip(curves.items(), colors):
        lw = 2.2 if c == BLUE else 1.4
        a1.plot(s.index, s.values, color=c, lw=lw, label=name)
        a1.annotate(f"{s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    a1.set_ylabel("Resultado acumulado neto, base 100")
    a1.set_title(title, loc="left", pad=28)
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    a2.fill_between(pos.index, 0, pos.values, step="post", color=BLUE, alpha=0.85, lw=0)
    a2.set_yticks([-3, -1, 0, 1, 3])
    a2.set_ylabel("Exposicion")
    a2.axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def full_history(curve: pd.Series, bh: pd.Series, oos_start, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvspan(pd.Timestamp(oos_start), curve.index[-1], color=BLUE, alpha=0.07, lw=0)
    ax.plot(curve.index, curve.values, color=BLUE, label="Estrategia (neta de carry y costos)")
    ax.plot(bh.index, bh.values, color=ORANGE, lw=1.4, label="Comprar siempre")
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.text(pd.Timestamp(oos_start), ax.get_ylim()[1], "  fuera de muestra", va="top", color=INK2, fontsize=9)
    ax.text(curve.index[0], ax.get_ylim()[1], "validacion walk-forward (desarrollo)", va="top", color=INK2, fontsize=9)
    ax.set_ylabel("Resultado acumulado, base 100")
    ax.set_title("Historia completa: desarrollo 2013-2022 y fuera de muestra 2023-2026", loc="left")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(p: pd.Series, y: pd.Series, path, title):
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    se = np.sqrt(g.y * (1 - g.y) / g.n)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    lo, hi = min(g.p.min(), (g.y - 2 * se).min()) - 0.01, max(g.p.max(), (g.y + 2 * se).max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="Calibracion perfecta")
    ax.errorbar(g.p, g.y, yerr=2 * se, fmt="o", color=BLUE, ms=7, capsize=3, lw=1.2, label="Deciles (+-2 e.e.)")
    ax.set_xlabel("Probabilidad predicha de alza (en exceso de carry)")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars(values: pd.Series, path, title, ylabel, fmt="{:.1f}", color=BLUE):
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [color if v >= 0 else RED for v in values.values]
    ax.bar([str(i) for i in values.index], values.values, color=cols, width=0.6)
    for i, v in enumerate(values.values):
        ax.annotate(fmt.format(v), (i, v), xytext=(0, 3 if v >= 0 else -12), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gate_panel(ic: pd.Series, gate: pd.Series, big_ic: float, path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(ic.index, ic.values, color=BLUE, lw=1.6, label="IC movil del modelo (250 jornadas)")
    off = gate == 0
    ax.fill_between(ic.index, ic.min(), ic.max(), where=off, color=RED, alpha=0.12, lw=0, step="post",
                    label="Compuerta cerrada: se abstiene")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.axhline(big_ic, color=YELLOW, lw=1, ls="--", label=f"IC > {big_ic}: habilita exposicion 3")
    ax.set_title("Compuerta de conviccion: el modelo vigila su propio desempeno", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def structural_break(ic_year: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    for c, col in zip(ic_year.columns, [BLUE, ORANGE, AQUA]):
        ax.plot(ic_year.index, ic_year[c].values * 100, marker="o", ms=4, color=col, lw=1.6, label=c)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_ylabel("IC de Spearman anual x 100")
    ax.set_title("Quiebre 2025: las senales de lead-lag cambian de signo", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit_panel(obs_corr: pd.DataFrame, flat_share: pd.Series, vol_year: pd.Series, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    a = axs[0]
    a.plot(obs_corr.index, obs_corr["mismo_dia"], color=ORANGE, marker="o", ms=3, lw=1.4, label="vs spot mismo dia")
    a.plot(obs_corr.index, obs_corr["dia_anterior"], color=BLUE, marker="o", ms=3, lw=1.6, label="vs spot dia anterior")
    a.set_title("Dolar observado = spot de t-1", loc="left")
    a.set_ylabel("Correlacion de variaciones")
    a.legend(loc="center right")
    a = axs[1]
    a.bar(flat_share.index.astype(str), flat_share.values * 100, color=BLUE, width=0.7)
    a.set_title("% dias con maximo = minimo", loc="left")
    a.set_xticks(range(0, len(flat_share), 6))
    a.set_xticklabels(flat_share.index.astype(str)[::6])
    a = axs[2]
    a.bar(vol_year.index.astype(str), vol_year.values, color=AQUA, width=0.7)
    a.axvline(list(vol_year.index).index(1999), color=INK2, ls="--", lw=1)
    a.text(list(vol_year.index).index(1999) + 0.5, vol_year.max() * 0.95, "libre flotacion", color=INK2, fontsize=9)
    a.set_title("Volatilidad anual del USD/CLP (%)", loc="left")
    a.set_xticks(range(0, len(vol_year), 6))
    a.set_xticklabels(vol_year.index.astype(str)[::6])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
