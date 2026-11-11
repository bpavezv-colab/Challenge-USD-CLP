"""Carry de mantener USD contra CLP entre el cierre t y el cierre t+1.

carry_t = devengo en pesos (t -> t+1) - devengo en dolares (t -> t+1)

* Pesos: Indice de Camara Promedio (CLICP). Es un indice de devengo: el
  cociente CLICP_{t+1} / CLICP_t - 1 es exactamente el interes ganado en el
  periodo, incluidos fines de semana y feriados, sin necesidad de dividir por
  dias. Antes de nov-2002 (o si falta) se usa la TPM nominal (CHOVCHOV) en
  base Act/360.
* Dolares: SOFR desde abr-2018 y Fed Funds efectiva antes, Act/360. El LIBOR
  esta descontinuado desde 2024-09 y no se usa.

Una posicion comprada en dolar paga el carry y una vendida lo recibe:
pnl_t+1 = pos_t * (r_{t+1} - carry_t).
"""
from __future__ import annotations

import pandas as pd

from . import data


def daily_carry(raw: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    days = pd.Series(cal, index=cal).diff().shift(-1).dt.days  # dias corridos t -> t+1

    clicp = data.align(data.col(raw, "CLICP Index"), cal, max_ffill=10)
    clp_from_index = clicp.shift(-1) / clicp - 1.0
    tpm = data.align(data.col(raw, "CHOVCHOV Index"), cal, max_ffill=10)
    clp_from_tpm = tpm / 100.0 * days / 360.0
    clp = clp_from_index.where(clp_from_index.notna(), clp_from_tpm)

    sofr = data.align(data.col(raw, "SOFRRATE Index"), cal, max_ffill=10)
    ff = data.align(data.col(raw, "FEDL01 Index"), cal, max_ffill=10)
    usd_rate = sofr.where(sofr.notna(), ff)
    usd = usd_rate / 100.0 * days / 360.0

    out = pd.DataFrame({"clp_accrual": clp, "usd_accrual": usd, "days": days})
    out["carry"] = out["clp_accrual"] - out["usd_accrual"]
    return out
