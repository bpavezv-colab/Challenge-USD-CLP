"""Seleccion de los parametros de la estrategia usando SOLO 2013-2022.

Uso:  python scripts/03_grilla_desarrollo.py

Reproduce la grilla con la que se eligieron los parametros de
``strategy.FINAL_PARAMS`` y la guarda en reports/tables/grilla_desarrollo.csv.
El tramo fuera de muestra no se usa aqui: las posiciones se recortan a
[VAL_START, DEV_END] antes de medir.

El numero total de ensayos (esta grilla mas las pruebas de modelo y de
variables descritas en el informe) alimenta el Deflated Sharpe Ratio.
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import pandas as pd

from usdclp import backtest as bt
from usdclp import config, features, strategy

df = features.build_dataset()
pred = pd.read_parquet(config.DATA_PROCESSED / "pred_wf.parquet")
ex = df["excess_next"]
DEV = slice(config.VAL_START, config.DEV_END)

rows = []
grid = itertools.product([1, 2, 3, 5, 8, 12], [0.0, 0.2, 0.3], [0.9, 0.95, 1.0], [None, 120, 250], [0.0])
grid = list(grid) + [(3, 0.2, q, 250, ic) for q in (0.7, 0.8, 0.9) for ic in (0.03, 0.05, 0.08, 0.10)]
for span, q1, bq, gw, bic in grid:
    prm = strategy.StrategyParams(smooth_span=span, q1=q1, big_q=bq, big_ic=bic if bic else -1.0, gate_window=gw)  # big_ic=-1: exposicion 3 solo por cuantil
    pos = strategy.positions(pred["p_cal"], ex, prm)["pos"].loc[DEV]
    r = bt.run(pos, ex)
    mt = bt.metrics(r)
    yr = r["net"].groupby(r.index.year).sum()
    rows.append({"smooth_span": span, "q1": q1, "big_q": bq, "big_ic": bic, "gate_window": gw,
                 "sharpe": mt["sharpe"], "retorno_anual_%": mt["retorno_anual_%"],
                 "max_drawdown_%": mt["max_drawdown_%"], "dias_activos_%": mt["dias_activos_%"],
                 "rotacion_anual": mt["rotacion_anual"], "anos_positivos": int((yr > 0).sum()),
                 "sharpe_bruto": bt.metrics(r, "gross")["sharpe"]})
R = pd.DataFrame(rows)
config.TABLES.mkdir(parents=True, exist_ok=True)
R.round(4).to_csv(config.TABLES / "grilla_desarrollo.csv", index=False)
print(f"{len(R)} configuraciones evaluadas en {config.VAL_START}..{config.DEV_END}")
print(R.sort_values("sharpe", ascending=False).head(10).round(3).to_string(index=False))
f = strategy.FINAL_PARAMS
sel = R[(R.smooth_span == f.smooth_span) & (R.q1 == f.q1) & (R.big_q == f.big_q) & (R.big_ic == f.big_ic) & (R.gate_window == f.gate_window)]
print("\nConfiguracion elegida:\n", sel.round(3).to_string(index=False))
