"""Funciones de alto nivel para el notebook de entrega y el monitor del
laboratorio: cargar o generar predicciones, construir posiciones, calcular las
estadisticas de trading pedidas y exportar las decisiones dia a dia."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config, features, pipeline, strategy

# version -> (modelo base, parametros de la regla)
VERSIONS = {
    "v2": ("reg", strategy.V2_PARAMS),        # produccion (monitoreo en vivo)
    "v1": ("clf", strategy.FINAL_PARAMS),     # oficial pre-registrada
}

PERIODS = {
    "Desarrollo 2013-2022": (config.VAL_START, config.DEV_END),
    "Fuera de muestra 2023-2026": (config.OOS_START, config.OOS_END),
    "Seguimiento oct-nov 2026": (config.FORWARD_START, None),
}


def get_predictions(df: pd.DataFrame, kind: str, recompute: bool = False) -> pd.DataFrame:
    """Predicciones walk-forward del modelo ``kind``. Si ya existen se
    extienden hasta la ultima fecha de la base; si no, se generan (~1-3 min)."""
    path = pipeline.pred_path(kind)
    existing = None if recompute or not path.exists() else pd.read_parquet(path)
    if existing is not None and existing.index.max() >= df.index.max():
        return existing
    pred = pipeline.update_predictions(df, existing, kind=kind)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path)
    return pred


def build_positions(df: pd.DataFrame, pred: pd.DataFrame, version: str) -> pd.DataFrame:
    _, prm = VERSIONS[version]
    return strategy.positions(pred["p_cal"], df["excess_next"], prm, components=pred, vol=df["clp_rv20"])


def reference_positions(P: pd.DataFrame) -> dict[str, pd.Series]:
    idx = P.index
    return {
        "Comprar siempre": pd.Series(1, index=idx),
        "No operar nunca": pd.Series(0, index=idx),
        "Exposicion uno (signo del modelo)": np.sign(P["p_model"] - 0.5).fillna(0).astype(int),
    }


def _slice(start, end, df):
    end = end or df["excess_next"].dropna().index.max()
    return slice(start, end)


def trading_stats(r: pd.DataFrame) -> dict:
    """Estadisticas pedidas por el laboratorio, netas de carry y costos."""
    n, g = bt.metrics(r, "net"), bt.metrics(r, "gross")
    return {
        "Jornadas": n["dias"],
        "Retorno neto total %": n["retorno_total_%"],
        "Retorno neto anual %": n["retorno_anual_%"],
        "Retorno bruto anual % (antes de costos)": g["retorno_anual_%"],
        "Volatilidad anual %": n["vol_anual_%"],
        "Sharpe neto": n["sharpe"],
        "Max drawdown %": n["max_drawdown_%"],
        "% aciertos (dias con posicion)": n["acierto_dias_activos_%"],
        "Numero de operaciones": n["operaciones"],
        "Dias con posicion": n["dias_con_posicion"],
        "Costos totales %": r["cost"].sum() * 100,
    }


def stats_table(df: pd.DataFrame, positions: dict[str, pd.Series], period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    sl = _slice(start, end, df)
    rows = {name: trading_stats(bt.run(pos.loc[sl], df["excess_next"])) for name, pos in positions.items()}
    return pd.DataFrame(rows).T


def motivo(row) -> str:
    if row["pos"] == 0:
        if row.get("vol_state", 0) == 2:
            return "fuera: volatilidad extrema (> p95)"
        if row["gate"] == 0:
            return "fuera: compuerta cerrada (IC del modelo <= 0)"
        return "fuera: conviccion bajo el umbral"
    if abs(row["pos"]) == 3:
        return "x3: conviccion en el 10% superior y IC > 0,05"
    if row.get("vol_state", 0) >= 1 and row.get("conv_big", False):
        return "x1: x3 bloqueado por volatilidad alta (> p80)"
    return "x1: conviccion sobre el umbral"


def export_positions(df: pd.DataFrame, P: pd.DataFrame, path) -> pd.DataFrame:
    """Decision de cada jornada, con todo lo necesario para replicarla."""
    conv_big = (P["edge"].abs() >= P["t_big"]) & (P["model_ic"] > strategy.V2_PARAMS.big_ic)
    Q = P.assign(conv_big=conv_big)
    r = bt.run(P["pos"], df["excess_next"])
    out = pd.DataFrame({
        "fecha_decision": P.index.date,
        "usdclp_cierre_t": df["px"].reindex(P.index).values,
        "prob_alza_modelo": P["p_model"].round(5).values,
        "prob_alza_suavizada": P["p"].round(5).values,
        "umbral_abstencion": P["t1"].round(5).values,
        "umbral_x3": P["t_big"].round(5).values,
        "ic_modelo_250d": P["model_ic"].round(4).values,
        "compuerta": P["gate"].values,
        "estado_volatilidad": P["vol_state"].values,
        "exposicion": P["pos"].values,
        "motivo": [motivo(row) for _, row in Q.iterrows()],
        "usdclp_cierre_t+1": df["px"].shift(-1).reindex(P.index).values,
        "carry_t_pb": (df["carry"].reindex(P.index) * 1e4).round(3).values,
        "resultado_neto_t+1_pb": (r["net"] * 1e4).round(2).values,
    })
    out.to_csv(path, index=False)
    return out


INT_ROWS = ("Jornadas", "Numero de operaciones", "Dias con posicion")


def show(table: pd.DataFrame) -> pd.DataFrame:
    """Tabla para mostrar: metricas en filas, estrategias en columnas; conteos
    como enteros y el resto con 2 decimales."""
    t = table.T.astype(object)
    for idx in t.index:
        t.loc[idx] = [("-" if pd.isna(v) else f"{v:,.0f}") if idx in INT_ROWS else ("-" if pd.isna(v) else f"{v:,.2f}")
                      for v in t.loc[idx]]
    return t
