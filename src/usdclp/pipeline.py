"""Orquestacion: dataset -> predicciones walk-forward -> calibracion.

Los hiperparametros elegidos en el desarrollo (2013-2022) quedan fijos aqui.
"""
from __future__ import annotations

import pandas as pd

from . import config, features, model

# Conjunto nucleo de variables, definido a priori por razonamiento economico
# (ver INFORME.md, seccion 3). No se selecciono mirando el tramo fuera de
# muestra.
CORE_FEATURES = [
    # Dinamica propia del USD/CLP
    "clp_r1", "clp_r5", "clp_z_r1", "clp_dist_ma20", "clp_close_loc", "clp_range", "clp_rv_ratio",
    # Monedas pares y dolar global (cierre posterior a Santiago)
    "peers_r1", "usdzar_r1", "usdmxn_r1", "usdbrl_r1", "bbdxy_r1", "resid_peers_r1",
    # Cobre y commodities
    "cu_comex_r1", "cu_lme_r1", "cu_comex_r5", "gold_r1",
    # Apetito por riesgo y volatilidad
    "vix_d1", "spx_r1", "emvxy_d1", "clpv1m_d1", "clp_vrp",
    # Tasas, carry y base NDF
    "diff_2y_d5", "ndf_basis", "ndf_pts_d1", "us2y_d1", "diff_on",
    # Flujos de la banca (rezago de publicacion de 2 dias)
    "flow_off_d1", "flow_off_d5",
    # Calendario
    "dow", "month_end",
]

MODEL_PARAMS = dict(C=0.005, use_gbm=True, n_seeds=3)
HALF_LIFE_YEARS = 3.0
REFIT_FREQ = "MS"
PRED_START = "2008-01-01"  # 3 anos de calentamiento para el calibrador


def model_factory(kind: str, feats: list[str], **mkw):
    """kind: 'clf' (v1: logit + GBM clasificador), 'reg' (ridge + GBM sobre
    retorno normalizado por vol), 'combo' (promedio de ambos), 'null'
    (clasificador con etiquetas barajadas)."""
    if kind == "clf":
        params = {**MODEL_PARAMS, **mkw}
        return lambda: model.Ensemble(feats, **params)
    if kind == "reg":
        return lambda: model.RegEnsemble(feats, n_seeds=MODEL_PARAMS["n_seeds"])
    if kind == "combo":
        return lambda: model.Combo(feats, C=MODEL_PARAMS["C"], n_seeds=MODEL_PARAMS["n_seeds"])
    if kind == "null":
        return lambda: model.ShuffledNull(model.Ensemble(feats, **MODEL_PARAMS))
    raise ValueError(kind)


def predict_walk_forward(df: pd.DataFrame, end: str, feats: list[str] = CORE_FEATURES,
                         start: str = PRED_START, half_life: float | None = HALF_LIFE_YEARS,
                         freq: str = REFIT_FREQ, kind: str = "clf", **mkw) -> pd.DataFrame:
    mask = features.training_mask(df)
    pred = model.walk_forward(df, mask, feats, start, end, model_factory(kind, feats, **mkw),
                              freq=freq, half_life_years=half_life)
    pred["p_cal"] = model.calibrate_walk_forward(pred, df["y"], mask)
    return pred


def fit_final(df: pd.DataFrame, asof: pd.Timestamp, feats: list[str] = CORE_FEATURES):
    """Modelo de produccion: entrenado con todo lo disponible hasta ``asof``."""
    mask = features.training_mask(df)
    tr = df.loc[config.TRAIN_START:asof]
    tr = tr[mask.reindex(tr.index).fillna(False)]
    w = model.time_decay_weights(tr.index, pd.Timestamp(asof), HALF_LIFE_YEARS)
    return model.Ensemble(feats, **MODEL_PARAMS).fit(tr[feats], tr["y"], sample_weight=w)


# Version de produccion (monitoreo en vivo): v2 = regresion sobre el retorno
# normalizado por volatilidad + gestion por volatilidad. Elegida solo con
# 2013-2022 (scripts/07_v2_seleccion.py). La v1 ("clf") sigue siendo la
# cifra oficial pre-registrada del fuera de muestra.
PRODUCTION_KIND = "reg"


def pred_path(kind: str):
    return config.DATA_PROCESSED / ("pred_wf.parquet" if kind == "clf" else f"pred_wf_{kind}.parquet")


def update_predictions(df: pd.DataFrame, existing: pd.DataFrame | None, kind: str = "clf") -> pd.DataFrame:
    """Extiende las predicciones walk-forward hasta la ultima fecha de ``df``.

    Produccion y backtest usan exactamente el mismo procedimiento: el mes en
    curso se predice con el modelo reentrenado al inicio del mes. Solo se
    recalculan los meses desde el ultimo reentrenamiento guardado.
    """
    end = df.index.max()
    if existing is None or existing.empty:
        return predict_walk_forward(df, end=str(end.date()), kind=kind)
    last_refit = pd.Timestamp(existing["refit"].max())
    keep = existing[existing["refit"] < last_refit]
    mask = features.training_mask(df)
    new = model.walk_forward(df, mask, CORE_FEATURES, str(last_refit.date()), str(end.date()),
                             model_factory(kind, CORE_FEATURES), freq=REFIT_FREQ,
                             half_life_years=HALF_LIFE_YEARS)
    out = pd.concat([keep.drop(columns=["p_cal"], errors="ignore"), new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["p_cal"] = model.calibrate_walk_forward(out, df["y"], mask)
    return out
