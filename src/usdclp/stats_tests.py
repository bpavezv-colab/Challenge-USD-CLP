"""Pruebas de significancia exigidas por las bases y controles de sobreajuste.

* Diebold-Mariano (1995) con varianza de largo plazo Newey-West y la
  correccion de muestra pequena de Harvey, Leybourne y Newbold (1997).
* Bootstrap estacionario de Politis y Romano (1994): remuestrea bloques de
  largo aleatorio (geometrico), preservando la autocorrelacion y la
  agrupacion de volatilidad de los retornos diarios.
* Probabilistic y Deflated Sharpe Ratio (Bailey y Lopez de Prado, 2012/2014):
  corrigen el Sharpe por no normalidad y por el numero de configuraciones
  probadas durante el desarrollo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_var(x: np.ndarray, lags: int | None = None) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    v = xc @ xc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (xc[k:] @ xc[:-k]) / n
    return v


def diebold_mariano(loss_model: pd.Series, loss_bench: pd.Series, h: int = 1) -> dict:
    """H0: igual perdida esperada. H1 (una cola): el modelo tiene MENOR perdida.

    d_t = L_bench - L_model, asi que DM > 0 favorece al modelo.
    """
    d = (loss_bench - loss_model).dropna().values
    n = len(d)
    lrv = newey_west_var(d)
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p_one = 1 - stats.t.cdf(stat, df=n - 1)
    return {"n": n, "dif_media": d.mean(), "DM": stat, "p_valor_una_cola": p_one,
            "p_valor_dos_colas": 2 * min(p_one, 1 - p_one)}


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_test(strat: pd.Series, bench: pd.Series, n_boot: int = 5000, mean_block: float = 10,
                   seed: int = 42) -> dict:
    """Bootstrap estacionario de la diferencia de retorno medio y de Sharpe.

    H0: la estrategia no supera a la referencia (diferencia <= 0). El p-valor
    es la fraccion de remuestreos, centrados bajo H0, con diferencia al menos
    tan grande como la observada.
    """
    df = pd.concat([strat, bench], axis=1).dropna()
    s, b = df.iloc[:, 0].values, df.iloc[:, 1].values
    d = s - b
    n = len(d)
    rng = np.random.default_rng(seed)

    def sharpe(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(252) if sd > 0 else 0.0

    obs_mean = d.mean()
    obs_sr_s = sharpe(s)
    obs_sr_diff = obs_sr_s - sharpe(b)
    boot_mean = np.empty(n_boot)
    boot_sr = np.empty(n_boot)
    boot_srd = np.empty(n_boot)
    for i in range(n_boot):
        ix = stationary_bootstrap_indices(n, mean_block, rng)
        boot_mean[i] = d[ix].mean()
        boot_sr[i] = sharpe(s[ix])
        boot_srd[i] = sharpe(s[ix]) - sharpe(b[ix])
    p_mean = np.mean(boot_mean - obs_mean >= obs_mean)
    p_srd = np.mean(boot_srd - obs_sr_diff >= obs_sr_diff)
    return {
        "n": n,
        "dif_retorno_anual_%": obs_mean * 252 * 100,
        "IC95_dif_retorno_anual_%": tuple(np.percentile(boot_mean, [2.5, 97.5]) * 252 * 100),
        "p_valor_dif_retorno": p_mean,
        "sharpe_estrategia": obs_sr_s,
        "IC95_sharpe": tuple(np.percentile(boot_sr, [2.5, 97.5])),
        "dif_sharpe": obs_sr_diff,
        "p_valor_dif_sharpe": p_srd,
    }


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: probabilidad de que el Sharpe verdadero supere ``sr_benchmark``
    (ambos en unidades por periodo, no anualizados)."""
    r = pd.Series(r).dropna()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_trials_ann_std: float) -> dict:
    """DSR: PSR contra el maximo Sharpe esperado de ``n_trials`` estrategias
    sin habilidad, dada la dispersion de Sharpes observada entre los ensayos."""
    emc = 0.5772156649
    sd = sr_trials_ann_std / np.sqrt(252)
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = sd * ((1 - emc) * z1 + emc * z2)
    return {"n_ensayos": n_trials, "sharpe_umbral_anual": sr0 * np.sqrt(252),
            "PSR_vs_0": probabilistic_sharpe(r, 0.0), "DSR": probabilistic_sharpe(r, sr0)}


def alpha_beta(strat: pd.Series, market: pd.Series) -> dict:
    """Regresion del resultado diario de la estrategia contra el de comprar
    siempre, con errores Newey-West. Separa cuanto del resultado es estar
    comprado o vendido en promedio (beta) y cuanto es timing (alfa)."""
    d = pd.concat([strat, market], axis=1).dropna()
    y, x = d.iloc[:, 0].values, d.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += w * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = b / se
    return {"alfa_anual_%": b[0] * 252 * 100, "t_alfa": t[0], "p_alfa_una_cola": 1 - stats.t.cdf(t[0], n - 2),
            "beta": b[1], "t_beta": t[1]}
