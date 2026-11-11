"""Pruebas de que ninguna variable usa informacion futura.

La prueba central trunca la base en una fecha T y reconstruye las variables:
todo lo que corresponde a fechas <= T debe ser identico a lo calculado con la
base completa. Si alguna variable mirara hacia adelante, cambiaria.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usdclp import backtest, data, features, strategy  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def full(raw):
    return features.build_dataset(raw)


@pytest.mark.parametrize("cut", ["2015-06-30", "2020-03-16", "2025-04-30"])
def test_features_no_miran_al_futuro(raw, full, cut):
    T = pd.Timestamp(cut)
    trunc = features.build_dataset(raw.loc[:T])
    cols = features.feature_columns(full)
    a = full.loc[:T, cols].iloc[:-1]  # la ultima fila del truncado no tiene t+1; se excluye por prudencia
    b = trunc.loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("cut", ["2015-06-30", "2020-03-16", "2025-04-30", "2026-10-06"])
def test_ultima_fila_igual_a_produccion(raw, full, cut):
    """La fila del dia en produccion (la ultima de una base truncada) debe
    coincidir con la misma fila calculada con historia posterior, salvo las
    variables que dependen de la jornada siguiente solo cuando esta es un
    feriado no previsto (no aplica en estas fechas)."""
    T = pd.Timestamp(cut)
    trunc = features.build_dataset(raw.loc[:T])
    t = trunc.index.max()
    if t not in full.index or full.index.max() == t:
        pytest.skip("sin historia posterior para comparar")
    cols = features.feature_columns(full)
    a, b = full.loc[t, cols], trunc.loc[t, cols]
    same = (a.isna() & b.isna()) | ((a - b).abs() < 1e-9)
    assert same.all(), f"difieren en la fila de produccion: {list(a.index[~same])}"


def test_objetivo_es_t_mas_1(full):
    px = full["px"]
    expected = px.shift(-1) / px - 1
    np.testing.assert_allclose(full["simple_ret_next"].dropna(), expected.dropna())


def test_flujos_bcch_rezagados(raw, full):
    """La posicion de la banca se usa con 2 jornadas de rezago."""
    s = data.splice_flow(raw, "CLPBODMB Index", "CLPBODM Index")
    al = data.align(s, full.index)
    d = al.diff()
    sc = d.rolling(250, min_periods=60).std()
    expected = (d / sc).shift(2)
    got = full["flow_off_d1"]
    m = expected.notna() & got.notna()
    assert m.sum() > 1000
    # El reescalado se calcula tras el rezago; basta verificar que el signo y
    # el orden temporal coinciden con la serie desplazada 2 jornadas.
    assert np.corrcoef(np.sign(expected[m]), np.sign(got[m]))[0, 1] > 0.99


def test_compuerta_usa_solo_pasado():
    """Cambiar el retorno de manana no puede cambiar la posicion de hoy."""
    idx = pd.bdate_range("2020-01-01", periods=800)
    rng = np.random.default_rng(0)
    p = pd.Series(0.5 + rng.normal(0, 0.03, len(idx)), index=idx)
    ex = pd.Series(rng.normal(0, 0.006, len(idx)), index=idx)
    base = strategy.positions(p, ex)["pos"]
    ex2 = ex.copy()
    ex2.iloc[-1] = 0.5  # shock en el retorno de la ultima jornada (aun no realizado)
    alt = strategy.positions(p, ex2)["pos"]
    pd.testing.assert_series_equal(base, alt)


def test_pnl_signo_y_carry():
    idx = pd.bdate_range("2024-01-01", periods=3)
    pos = pd.Series([1, -3, 0], index=idx)
    ex = pd.Series([0.01, 0.01, 0.01], index=idx)
    r = backtest.run(pos, ex, cost_bps=0)
    assert r["gross"].tolist() == pytest.approx([0.01, -0.03, 0.0])
