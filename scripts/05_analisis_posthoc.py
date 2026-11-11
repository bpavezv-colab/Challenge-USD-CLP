"""Analisis POST-HOC: disenado DESPUES de mirar el fuera de muestra.

Uso:  python scripts/05_analisis_posthoc.py [--regenerar]

Nada de lo que hay aqui cambia la estrategia oficial ni sus cifras. Se deja
en el repositorio por transparencia (Lopez de Prado: reportar todos los
ensayos) y para explicar al jurado que aprendimos del tramo 2025-2026.

1. Diagnostico de 2026: sesgo direccional persistente de la senal.
2. Variante "sin sesgo": a la ventaja se le resta su media movil (apuesta solo
   al timing diario).
3. Memorias mas cortas (vida media 1 y 0.5 anos) y combinaciones ponderadas por
   desempeno reciente (Pesaran-Timmermann): no mejoran y se descartan.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from usdclp import backtest as bt
from usdclp import config, features, pipeline, strategy

df = features.build_dataset()
ex, y, mask = df["excess_next"], df["y"], features.training_mask(df)
last = ex.dropna().index.max()
PER = {"DEV": slice(config.VAL_START, config.DEV_END), "OOS": slice(config.OOS_START, last)}

base = pd.read_parquet(config.DATA_PROCESSED / "pred_wf.parquet")["p_cal"]
variants = {"Oficial (v1, vida media 3 anos)": base}

for hl in (1.0, 0.5):
    f = config.DATA_PROCESSED / f"pred_wf_hl{hl}.parquet"
    if "--regenerar" in sys.argv or not f.exists():
        pipeline.predict_walk_forward(df, end=str(df.index.max().date()), half_life=hl).to_parquet(f)
    variants[f"Vida media {hl:g} anos"] = pd.read_parquet(f)["p_cal"]

idx = base.index
mem = {k: v.reindex(idx) for k, v in variants.items()}
loss = pd.DataFrame({k: ((v - y.reindex(idx)) ** 2).where(mask.reindex(idx)) for k, v in mem.items()}).shift(1)
best = loss.rolling(120, min_periods=60).mean().dropna(how="all").idxmin(axis=1).reindex(idx).fillna(list(mem)[0])
variants["Combinacion: mejor Brier ultimas 120 jornadas"] = pd.Series([mem[b][t] for t, b in best.items()], index=idx)
variants["Combinacion: promedio simple"] = pd.DataFrame(mem).mean(axis=1)
for w in (60, 250):
    variants[f"Sin sesgo direccional (resta media movil {w})"] = 0.5 + (base - 0.5) - (base - 0.5).shift(1).rolling(w, min_periods=w // 2).mean()

rows = {}
for name, p in variants.items():
    row = {}
    for per, sl in PER.items():
        P = strategy.positions(p, ex)
        r = bt.run(P["pos"].loc[sl], ex)
        m = bt.metrics(r)
        row[f"{per} retorno neto anual %"] = m["retorno_anual_%"]
        row[f"{per} Sharpe neto"] = m["sharpe"]
        row[f"{per} max DD %"] = m["max_drawdown_%"]
        row[f"{per} % dias vendido"] = (P["pos"].loc[sl] < 0).mean() * 100
    rows[name] = row
T = pd.DataFrame(rows).T
T.round(2).to_csv(config.TABLES / "posthoc_variantes.csv")
print(T.round(2).to_string())

# Diagnostico 2023-2026: sesgo de la senal vs tendencia del dolar
P = strategy.positions(base, ex)
diag = {}
for yr in range(2023, last.year + 1):
    d = pd.DataFrame({"edge": P["edge"], "x": ex, "pos": P["pos"]}).loc[str(yr)].dropna()
    diag[yr] = {"IC diario senal": d.edge.corr(d.x, method="spearman"),
                "% dias vendido": (d.pos < 0).mean() * 100, "% dias comprado": (d.pos > 0).mean() * 100,
                "exceso USD/CLP del ano %": d.x.sum() * 100}
D = pd.DataFrame(diag).T
D.round(3).to_csv(config.TABLES / "posthoc_diagnostico_sesgo.csv")
print("\n", D.round(2).to_string())
