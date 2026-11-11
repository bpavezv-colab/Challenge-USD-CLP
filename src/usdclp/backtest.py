"""Motor de backtest: exposicion x (variacion USDCLP - carry) - costos.

Convenciones (las mismas que describen las bases):

* La posicion ``pos_t`` se decide al cierre de t con informacion hasta t y se
  mantiene hasta el cierre de t+1.
* Retorno de la jornada t+1 por unidad de nocional:
      pnl_{t+1} = pos_t * (S_{t+1}/S_t - 1 - carry_t) - c * |pos_t - pos_{t-1}|
  donde carry_t es devengo CLP - devengo USD entre t y t+1. Comprado en dolar
  paga el carry; vendido lo recibe. Posicion cero = en pesos ganando la tasa
  local, que es justamente la referencia del exceso.
* El nocional es fijo, asi que el resultado acumulado es la suma de los
  retornos diarios (base 100 = 100 + 100 * suma).

Si el codigo oficial del laboratorio usa otra convencion, basta reemplazar
``run`` manteniendo la firma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def positions_from_proba(p: pd.Series, t1: float, t2: float, big: int = 3) -> pd.Series:
    """Mapea probabilidad a exposicion en {-3,-1,0,1,3}.

    |p - 0.5| < t1          -> 0   (abstenerse)
    t1 <= |p - 0.5| < t2    -> +-1
    |p - 0.5| >= t2         -> +-3
    """
    e = p - 0.5
    a = e.abs()
    size = np.where(a < t1, 0, np.where(a < t2, 1, big))
    return pd.Series(np.sign(e) * size, index=p.index).fillna(0).astype(int)


def run(pos: pd.Series, excess_next: pd.Series, cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    pos = pos.astype(float).fillna(0.0)
    ex = excess_next.reindex(pos.index)
    trade = pos.diff().abs()
    trade.iloc[0] = abs(pos.iloc[0])
    gross = pos * ex
    cost = trade * cost_bps / 1e4
    out = pd.DataFrame({"pos": pos, "gross": gross, "cost": cost, "net": gross - cost, "trade": trade})
    # Ultima fila: decision para manana, retorno aun desconocido.
    return out


def metrics(bt: pd.DataFrame, col: str = "net") -> dict:
    r = bt[col].dropna()
    n = len(r)
    if n == 0:
        return {}
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    eq = r.cumsum()
    dd = eq - eq.cummax()
    active = bt.loc[r.index, "pos"] != 0
    # Una operacion = una jornada en que la exposicion cambia (entrar, salir,
    # invertir o cambiar de tamano). Cada operacion paga costo.
    n_ops = int((bt.loc[r.index, "trade"] > 0).sum())
    return {
        "dias": n,
        "retorno_total_%": r.sum() * 100,
        "retorno_anual_%": ann * 100,
        "vol_anual_%": vol * 100,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "max_drawdown_%": dd.min() * 100,
        "calmar": (ann / -dd.min()) if dd.min() < 0 else np.nan,
        "acierto_dias_activos_%": (bt.loc[r.index, "gross"][active] > 0).mean() * 100 if active.any() else np.nan,
        "dias_activos_%": active.mean() * 100,
        "dias_con_posicion": int(active.sum()),
        "operaciones": n_ops,
        "operaciones_por_ano": n_ops / n * 252,
        "exposicion_media_abs": bt.loc[r.index, "pos"].abs().mean(),
        "rotacion_anual": bt.loc[r.index, "trade"].sum() / n * 252,
        "costos_anual_%": bt.loc[r.index, "cost"].sum() / n * 252 * 100,
    }


def equity_curve(bt: pd.DataFrame, col: str = "net") -> pd.Series:
    """Resultado acumulado base 100, fechado en el dia de realizacion (t+1)."""
    r = bt[col].dropna()
    return 100 + 100 * r.cumsum()
