"""Construccion de variables explicativas y del objetivo.

Toda variable de la fila t usa solo informacion disponible al cierre de la
jornada t (supuesto de las bases: el modelo corre al cierre de cada sesion y
los cierres de una misma fecha estan disponibles al decidir). El objetivo de
la fila t es el retorno en exceso de t a t+1.

``strict_timing=True`` rezaga un dia todas las series que cierran despues de
Santiago (Nueva York, Londres). Sirve como prueba de robustez: si el modelo
solo funcionara gracias a esa ventana horaria, se veria aqui.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import carry as carry_mod
from . import config, data

# Series que cierran en Santiago o antes (no se rezagan en modo estricto).
SANTIAGO_OR_EARLIER = {
    "USDCLP Curncy", "CLICP Index", "CHOVCHOV Index", "CHSWP1 Curncy", "CHSWP2 Curncy",
    "CHSWP5 Curncy", "CHSWP10 Curncy", "CHSWC2 Curncy", "SCO1 Comdty", "SHCOMP Index",
    "USDCNY Curncy", "LMCADS03 Comdty", "LMAHDS03 Comdty", "CLPBODMB Index", "CLPBSMB Index",
    "CLPBLDMB Index",
}

PRICE_SERIES = {
    # nombre corto: ticker
    "cu_lme": "LMCADS03 Comdty",
    "cu_comex": "HG1 Comdty",
    "alu": "LMAHDS03 Comdty",
    "brent": "CO1 Comdty",
    "gold": "XAU Curncy",
    "bcom": "BCOM Index",
    "iron": "SCO1 Comdty",
    "dxy": "DXY Curncy",
    "bbdxy": "BBDXY Index",
    "eurusd": "EURUSD Curncy",
    "usdjpy": "USDJPY Curncy",
    "usdcnh": "USDCNH Curncy",
    "usdbrl": "USDBRL Curncy",
    "usdmxn": "USDMXN Curncy",
    "usdcop": "USDCOP Curncy",
    "usdpen": "USDPEN Curncy",
    "usdzar": "USDZAR Curncy",
    "spx": "SPX Index",
    "mxef": "MXEF Index",
    "shcomp": "SHCOMP Index",
}

LEVEL_SERIES = {
    "vix": "VIX Index",
    "move": "MOVE Index",
    "emvxy": "JPMVXYEM Index",
    "clpv1m": "USDCLPV1M Curncy",
    "clpv3m": "USDCLPV3M Curncy",
    "us3m": "USGG3M Index",
    "us2y": "USGG2YR Index",
    "us10y": "USGG10YR Index",
    "sw1": "CHSWP1 Curncy",
    "sw2": "CHSWP2 Curncy",
    "sw5": "CHSWP5 Curncy",
    "sw10": "CHSWP10 Curncy",
    "swuf2": "CHSWC2 Curncy",
    "tpm": "CHOVCHOV Index",
    "ndf_y1m": "CHNI1M Curncy",
    "ndf_y3m": "CHNI3M Curncy",
    "ndf_p1m": "CHN1M Curncy",
}


def _lag_for(ticker: str, strict: bool) -> int:
    lag = config.PUBLICATION_LAG.get(ticker, 0)
    if strict and ticker not in SANTIAGO_OR_EARLIER:
        lag += 1
    return lag


def _z(x: pd.Series, w: int) -> pd.Series:
    m = x.rolling(w, min_periods=w // 2).mean()
    s = x.rolling(w, min_periods=w // 2).std()
    return (x - m) / s


def build_dataset(raw: pd.DataFrame | None = None, strict_timing: bool = False) -> pd.DataFrame:
    """Devuelve un DataFrame con features, objetivo y columnas auxiliares.

    Columnas auxiliares (no son features): ``px``, ``ret_next`` (retorno
    log t->t+1), ``carry`` (carry t->t+1), ``excess_next``, ``y`` (1 si el
    exceso es positivo), ``stale_next``, ``gap_next``.
    """
    raw = data.load_raw() if raw is None else raw
    cal = data.trading_calendar(raw)
    F = pd.DataFrame(index=cal)

    # ---------- Serie objetivo -------------------------------------------
    px = data.col(raw, config.TARGET).reindex(cal)
    hi = data.col(raw, config.TARGET, "PX_HIGH").reindex(cal)
    lo = data.col(raw, config.TARGET, "PX_LOW").reindex(cal)
    lp = np.log(px)
    r = lp.diff()
    for k in (1, 2, 5, 10, 20, 60):
        F[f"clp_r{k}"] = lp.diff(k)
    rv20 = r.rolling(20).std()
    rv60 = r.rolling(60).std()
    F["clp_rv20"] = rv20 * np.sqrt(252)
    F["clp_rv_ratio"] = r.rolling(5).std() / rv60
    for w in (20, 100):
        F[f"clp_dist_ma{w}"] = (lp - lp.rolling(w).mean()) / (rv20 * np.sqrt(w))
    F["clp_z_r1"] = r / rv20
    # Rango intradia: solo tiene contenido desde 2004 (antes H=L=C).
    rng = np.log(hi / lo)
    rng[rng <= 0] = np.nan
    F["clp_range"] = rng / rv20
    F["clp_close_loc"] = ((px - lo) / (hi - lo)).where(hi > lo)
    F["clp_park20"] = np.sqrt((rng**2).rolling(20, min_periods=10).mean() / (4 * np.log(2))) * np.sqrt(252)
    F["clp_rsi14"] = _rsi(px, 14)

    # ---------- Precios externos: retornos -------------------------------
    rets = {}
    for name, tk in PRICE_SERIES.items():
        s = data.align(data.col(raw, tk), cal, lag=_lag_for(tk, strict_timing))
        s = s.where(s > 0)  # WTI negativo no se usa; resto siempre > 0
        ls = np.log(s)
        rets[name] = ls.diff()
        F[f"{name}_r1"] = ls.diff()
        F[f"{name}_r5"] = ls.diff(5)
        F[f"{name}_r20"] = ls.diff(20)
    R = pd.DataFrame(rets)

    # Canasta de monedas pares (todas como USD/xxx: sube = dolar mas fuerte)
    peers = R[["usdbrl", "usdmxn", "usdcop", "usdpen", "usdzar"]]
    F["peers_r1"] = peers.mean(axis=1)
    F["peers_r5"] = peers.rolling(5).sum().mean(axis=1)
    # Desvio del CLP contra sus pares y contra el cobre (beta movil 250d)
    for name, x in (("peers", F["peers_r1"]), ("cu", R["cu_lme"]), ("dxy", R["dxy"])):
        cov = r.rolling(250, min_periods=120).cov(x)
        var = x.rolling(250, min_periods=120).var()
        beta = cov / var
        F[f"beta_{name}"] = beta
        resid = r - beta * x
        F[f"resid_{name}_r1"] = resid / rv20
        F[f"resid_{name}_r5"] = resid.rolling(5).sum() / (rv20 * np.sqrt(5))
    # Cobre en pesos: cuanto se movio el cobre medido en CLP
    F["cu_clp_r5"] = F["cu_lme_r5"] + F["clp_r5"]

    # ---------- Niveles: volatilidad, tasas -------------------------------
    L = {}
    for name, tk in LEVEL_SERIES.items():
        L[name] = data.align(data.col(raw, tk), cal, lag=_lag_for(tk, strict_timing))
    L = pd.DataFrame(L)
    for v in ("vix", "move", "emvxy", "clpv1m"):
        F[f"{v}_lvl_z"] = _z(L[v], 250)
        F[f"{v}_d1"] = L[v].diff()
        F[f"{v}_d5"] = L[v].diff(5)
    F["clp_vrp"] = L["clpv1m"] - F["clp_rv20"] * 100
    F["clp_vol_slope"] = L["clpv3m"] - L["clpv1m"]
    for t in ("us2y", "us10y", "sw2", "sw10"):
        F[f"{t}_d1"] = L[t].diff()
        F[f"{t}_d5"] = L[t].diff(5)
    F["diff_2y"] = L["sw2"] - L["us2y"]
    F["diff_2y_d5"] = F["diff_2y"].diff(5)
    F["diff_2y_d20"] = F["diff_2y"].diff(20)
    F["clp_curve"] = L["sw10"] - L["sw1"]
    F["us_curve"] = L["us10y"] - L["us3m"]
    F["breakeven2y_d5"] = (L["sw2"] - L["swuf2"]).diff(5)

    usd_on = data.align(data.col(raw, "SOFRRATE Index"), cal, lag=_lag_for("SOFRRATE Index", strict_timing))
    ff_on = data.align(data.col(raw, "FEDL01 Index"), cal, lag=_lag_for("FEDL01 Index", strict_timing))
    usd_on = usd_on.where(usd_on.notna(), ff_on)
    F["diff_on"] = L["tpm"] - usd_on
    # Base onshore/offshore: rendimiento implicito NDF 1M menos la tasa local
    F["ndf_basis"] = L["ndf_y1m"] - L["tpm"]
    F["ndf_basis_d5"] = F["ndf_basis"].diff(5)
    F["ndf_pts_d1"] = L["ndf_p1m"].diff() / px * 1e4

    # ---------- Flujos de la banca (BCCh, rezago de publicacion) ---------
    flows = {
        "flow_off": data.splice_flow(raw, "CLPBODMB Index", "CLPBODM Index"),
        "flow_loc": data.splice_flow(raw, "CLPBLDMB Index", "CLPBLDM Index"),
    }
    for name, s in flows.items():
        a = data.align(s, cal, lag=config.PUBLICATION_LAG["CLPBODMB Index"])
        sc = a.diff().rolling(250, min_periods=60).std()
        F[f"{name}_d1"] = a.diff() / sc
        F[f"{name}_d5"] = a.diff(5) / sc
        F[f"{name}_z"] = _z(a, 250)

    # ---------- Macro chilena point-in-time -------------------------------
    ipc = data.macro_point_in_time(raw, "CLINNSYO Index", cal)
    ima = data.macro_point_in_time(raw, "CLIMYOYN Index", cal)
    F["ipc_yoy"] = ipc
    F["imacec_yoy"] = ima
    F["real_tpm"] = L["tpm"] - ipc

    # ---------- Calendario -------------------------------------------------
    F["dow"] = cal.dayofweek
    # Proxima jornada: la observada en el calendario; para la ultima fila (el
    # dia en produccion) aun no existe y se usa el siguiente dia habil, que se
    # conoce de antemano. Sin esto la ultima fila quedaria siempre marcada
    # como fin de mes.
    nxt = pd.Series(cal, index=cal).shift(-1)
    nxt = nxt.fillna(pd.Series(cal + pd.offsets.BDay(1), index=cal))
    F["month_end"] = (nxt.dt.month != cal.month).astype(float)

    # ---------- Objetivo y auxiliares -------------------------------------
    c = carry_mod.daily_carry(raw, cal)
    stale = data.stale_target_mask(raw).reindex(cal).fillna(False)
    out = F.copy().replace([np.inf, -np.inf], np.nan)
    out["px"] = px
    out["ret_next"] = lp.shift(-1) - lp
    out["simple_ret_next"] = px.shift(-1) / px - 1
    out["carry"] = c["carry"]
    out["excess_next"] = out["simple_ret_next"] - out["carry"]
    out["y"] = (out["excess_next"] > 0).astype(float).where(out["excess_next"].notna())
    out["stale_next"] = stale.shift(-1).fillna(False).astype(bool)
    out["gap_next"] = c["days"]
    out = add_adaptive_features(out)
    return out


ADAPTIVE_SIGNALS = ["peers_r1", "usdzar_r1", "usdmxn_r1", "usdbrl_r1", "bbdxy_r1", "vix_d1",
                    "clp_z_r1", "cu_comex_r1", "spx_r1", "resid_peers_r1", "clp_close_loc"]
ADAPTIVE_WINDOWS = (120, 250)


def add_adaptive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coeficientes de lead-lag que se re-estiman cada dia.

    Para la senal x y el retorno en exceso e (de t a t+1), al cierre de t solo
    se conocen los pares (x_s, e_s) con s <= t-1. Se estima la pendiente movil
    b_t = cov(x, e) / var(x) sobre esos pares y se construye la prediccion
    univariada b_t * x_t, normalizada por la volatilidad. Asi el modelo puede
    detectar que una relacion cambio de signo (como ocurre desde 2025 con los
    pares y la autocorrelacion del CLP) sin esperar a reentrenarse.
    """
    e = (df["excess_next"] / (df["clp_rv20"] / np.sqrt(252))).shift(1)  # realizado al cierre de t
    total = {w: 0.0 for w in ADAPTIVE_WINDOWS}
    new = {}
    for sig in ADAPTIVE_SIGNALS:
        x = df[sig]
        xs = x.shift(1)
        for w in ADAPTIVE_WINDOWS:
            cov = xs.rolling(w, min_periods=w // 2).cov(e)
            var = xs.rolling(w, min_periods=w // 2).var()
            b = cov / var
            corr = xs.rolling(w, min_periods=w // 2).corr(e)
            new[f"ll{w}_corr_{sig}"] = corr
            pred = (b * x).clip(-3, 3)
            new[f"ll{w}_pred_{sig}"] = pred
            total[w] = total[w] + pred.fillna(0)
    for w in ADAPTIVE_WINDOWS:
        new[f"ll{w}_pred_sum"] = total[w] / len(ADAPTIVE_SIGNALS)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _rsi(px: pd.Series, n: int) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


AUX_COLS = ["px", "ret_next", "simple_ret_next", "carry", "excess_next", "y", "stale_next", "gap_next"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in AUX_COLS]


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Filas validas para entrenar: objetivo definido, sin cotizacion rancia en
    t+1 y sin saltos de calendario anomalos (hueco 2011-2012 del USDCLP)."""
    return df["y"].notna() & ~df["stale_next"] & (df["gap_next"] <= 5)
