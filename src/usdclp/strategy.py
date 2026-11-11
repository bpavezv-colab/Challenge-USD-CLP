"""De probabilidad a exposicion: suavizado, compuerta de conviccion y tamano.

Capas, cada una con su justificacion (INFORME.md, seccion 4):

1. Probabilidad calibrada ``p`` (ensamble + temperature scaling walk-forward).
2. Suavizado: la ventaja diaria (p - 0.5) se promedia con una EMA corta. La
   senal de un dia es ruidosa y cada cambio de posicion paga costos; el
   suavizado reduce la rotacion a la mitad sin perder la ventaja bruta.
3. Compuerta de conviccion (meta-labeling, Lopez de Prado cap. 3): si la
   correlacion entre la ventaja predicha y el exceso realizado en las ultimas
   ``gate_window`` jornadas no es positiva, el modelo no esta funcionando en
   el regimen actual y la estrategia se abstiene.
4. Tamano:
     |ventaja| bajo su cuantil q1 movil              -> 0  (abstenerse)
     sobre q1                                         -> +-1
     sobre el cuantil big_q y con IC reciente > big_ic -> +-3
   La exposicion maxima exige dos cosas: conviccion alta hoy Y evidencia de
   que el modelo ha acertado ultimamente.

Todos los umbrales usan solo informacion anterior a la decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    """Parametros elegidos en el desarrollo 2013-2022 (ver reports/tables)."""

    smooth_span: int = 3
    q1: float = 0.20
    big_q: float = 0.90
    big_ic: float = 0.05
    size_lookback: int = 250
    gate_window: int | None = 250
    gate_min: float = 0.0
    # --- Extensiones v2 (apagadas por defecto = v1) ---
    # Filtro de acuerdo: abstenerse si los modelos lineales y los de arboles
    # (suavizados) apuntan en direcciones opuestas.
    agree_filter: bool = False
    # Gestion por volatilidad (Moreira y Muir, 2017): si la volatilidad
    # realizada de 20 dias supera su cuantil movil ``vol_cap_q`` de 250
    # jornadas, la exposicion maxima es 1; sobre ``vol_off_q``, se abstiene.
    vol_cap_q: float | None = None
    vol_off_q: float | None = None

    def to_dict(self):
        return asdict(self)


FINAL_PARAMS = StrategyParams()  # v1 (oficial, pre-registrada)
# v2 (produccion): misma regla + no triplicar sobre el p80 de la volatilidad
# realizada de 20 dias y abstenerse sobre el p95 (scripts/07_v2_seleccion.py).
V2_PARAMS = StrategyParams(vol_cap_q=0.80, vol_off_q=0.95)


def smooth(p: pd.Series, span: int) -> pd.Series:
    if span <= 1:
        return p
    return 0.5 + (p - 0.5).ewm(span=span, adjust=False).mean()


def model_ic(p: pd.Series, excess_next: pd.Series, window: int) -> pd.Series:
    """Correlacion movil entre la ventaja predicha y el exceso realizado.

    El par (p_s, exceso_s) se conoce recien al cierre de s+1; por eso ambos se
    desplazan un dia antes de calcular la ventana. El exceso se estandariza y
    acota a +-3 desvios para que un par de jornadas extremas no dominen.
    """
    edge = (p - 0.5).shift(1)
    realized = excess_next.reindex(p.index).shift(1)
    vol = realized.rolling(60, min_periods=20).std()
    z = (realized / vol).clip(-3, 3)
    return edge.rolling(window, min_periods=window // 2).corr(z)


LINEAR_COMPONENTS = ("p_logit", "p_ridge")
TREE_COMPONENTS = ("p_gbm", "p_gbmr")


def positions(p_model: pd.Series, excess_next: pd.Series, prm: StrategyParams = FINAL_PARAMS,
              components: pd.DataFrame | None = None, vol: pd.Series | None = None) -> pd.DataFrame:
    p = smooth(p_model, prm.smooth_span)
    e = p - 0.5
    conv = e.abs()
    past_conv = conv.shift(1).rolling(prm.size_lookback, min_periods=prm.size_lookback // 2)
    t1 = past_conv.quantile(prm.q1)
    t_big = past_conv.quantile(prm.big_q) if prm.big_q < 1 else pd.Series(np.inf, index=p.index)

    # El IC del modelo se calcula siempre (lo usa la regla de exposicion 3);
    # la compuerta solo se aplica si gate_window esta definido.
    ic = model_ic(p, excess_next, prm.gate_window or 250)
    if prm.gate_window:
        gate = ((ic > prm.gate_min) | ic.isna()).astype(int)
    else:
        gate = pd.Series(1, index=p.index)

    size = np.where(t1.isna() | (conv < t1), 0, 1)
    big = (conv >= t_big) & (ic > prm.big_ic)
    size = np.where(big & (size > 0), 3, size)
    agree = pd.Series(1, index=p.index)
    if prm.agree_filter and components is not None:
        lin = [c for c in LINEAR_COMPONENTS if c in components]
        tre = [c for c in TREE_COMPONENTS if c in components]
        el = smooth(components[lin].mean(axis=1), prm.smooth_span) - 0.5
        et = smooth(components[tre].mean(axis=1), prm.smooth_span) - 0.5
        agree = ((np.sign(el) == np.sign(et)) | el.isna() | et.isna()).astype(int).reindex(p.index).fillna(1).astype(int)
        size = np.where(agree.values == 1, size, 0)

    vol_state = pd.Series(0, index=p.index)
    if vol is not None and (prm.vol_cap_q or prm.vol_off_q):
        v = vol.reindex(p.index)
        past_v = v.shift(1).rolling(250, min_periods=125)
        if prm.vol_cap_q:
            cap = (v > past_v.quantile(prm.vol_cap_q)).fillna(False)
            size = np.where(cap.values & (size == 3), 1, size)
            vol_state[cap] = 1
        if prm.vol_off_q:
            off = (v > past_v.quantile(prm.vol_off_q)).fillna(False)
            size = np.where(off.values, 0, size)
            vol_state[off] = 2

    pos = (np.sign(e).fillna(0) * size * gate).astype(int)

    return pd.DataFrame({
        "p_model": p_model, "p": p, "edge": e, "t1": t1, "t_big": t_big,
        "model_ic": ic, "gate": gate, "agree": agree, "vol_state": vol_state, "pos": pos,
    })
