"""Carga de la base Bloomberg y tratamiento point-in-time.

Reglas que aplica este modulo (ver INFORME.md, seccion 2):

* El calendario de trabajo es el de USDCLP Curncy (las jornadas en que cotiza
  la serie objetivo). El resto de las series se alinea a ese calendario.
* Un precio se arrastra hacia adelante como maximo ``MAX_FFILL`` jornadas. Si
  una serie deja de publicarse (LIBOR, IPSA rancio) queda vacia en vez de
  fabricar un dato viejo que el modelo tomaria por nuevo.
* Las series con rezago de publicacion se desplazan segun
  ``config.PUBLICATION_LAG``.
* Las series macro se ubican en la fecha de su publicacion (FECHA_RELEASE), no
  en la del mes al que se refieren.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MAX_FFILL = 5


def load_raw(refresh: bool = False, updates: bool = True) -> pd.DataFrame:
    """Devuelve la base ancha (fecha x 'ticker|campo').

    Parte de la base original del laboratorio y, si ``updates``, le aplica en
    orden las actualizaciones descargadas con la plantilla Bloomberg que haya
    en ``data/raw/actualizaciones/`` (ver ``parse_bloomberg_template``).
    """
    if config.RAW_PARQUET.exists() and not refresh:
        df = pd.read_parquet(config.RAW_PARQUET)
    else:
        df = pd.read_excel(config.RAW_XLSX, sheet_name="datos", index_col=0, parse_dates=True)
        df.to_parquet(config.RAW_PARQUET)
    df.index = pd.to_datetime(df.index)
    df.index.name = "fecha"
    df = df.sort_index()
    if updates and config.UPDATES_DIR.exists():
        for f in sorted(config.UPDATES_DIR.glob("actualizacion_hasta_*.xlsx")):
            hasta = pd.Timestamp(f.stem.split("_")[-1])
            df = merge_update(df, parse_bloomberg_template(f, hasta))
    return df


def parse_bloomberg_template(path, hasta: pd.Timestamp) -> pd.DataFrame:
    """Lee la hoja 'descarga' de la plantilla BDH (pares fecha/valor por serie).

    * Los errores de Bloomberg ('#N/A N/A', '#N/A Invalid Security', ...) quedan
      vacios.
    * FECHA_RELEASE llega como entero AAAAMMDD y se convierte a fecha.
    * Se descartan las fechas posteriores a ``hasta``: la ultima jornada
      descargada antes del cierre de Nueva York trae precios intradia, no
      cierres. La UF (CHUF) se publica por adelantado y tampoco se necesita.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb["descarga"].iter_rows(values_only=True))
    header = rows[0]
    out = {}
    for j in range(0, len(header), 2):
        name = header[j]
        if not name:
            continue
        dates, vals = [], []
        for r in rows[1:]:
            if j + 1 >= len(r) or r[j] is None or isinstance(r[j], str):
                continue
            v = r[j + 1]
            if isinstance(v, str):
                v = np.nan
            dates.append(pd.Timestamp(r[j]))
            vals.append(v)
        s = pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        if name.endswith("|FECHA_RELEASE"):
            s = pd.Series(pd.to_datetime(s.dropna().astype("int64").astype(str), format="%Y%m%d"),
                          index=s.dropna().index)
        out[name] = s
    new = pd.DataFrame(out).sort_index()
    new = new[new.index <= hasta]
    new.index.name = "fecha"
    return new


def merge_update(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Los valores nuevos prevalecen; donde la actualizacion viene vacia se
    conserva la base. En el traslape 1-sep a 6-oct-2026 ambas coinciden
    exactamente en todas las series de mercado."""
    new = new.reindex(columns=base.columns)
    # Un valor macro no publicado (#N/A) no debe borrar su FECHA_RELEASE vieja
    merged = new.combine_first(base)
    for c in base.columns:
        if c.endswith("|FECHA_RELEASE"):
            merged[c] = pd.to_datetime(merged[c])
    return merged.sort_index()


def col(df: pd.DataFrame, ticker: str, field: str = "PX_LAST") -> pd.Series:
    return df[f"{ticker}|{field}"]


def trading_calendar(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Jornadas en que cotiza el USD/CLP."""
    return col(raw, config.TARGET).dropna().index


def align(series: pd.Series, cal: pd.DatetimeIndex, lag: int = 0, max_ffill: int = MAX_FFILL) -> pd.Series:
    """Lleva una serie nativa al calendario de trabajo sin fabricar datos.

    1. Reindexa sobre la union de fechas y arrastra el ultimo dato conocido
       como maximo ``max_ffill`` jornadas nativas.
    2. Toma las fechas del calendario de trabajo.
    3. Aplica el rezago de publicacion (en jornadas del calendario).
    """
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index=cal, name=series.name)
    union = s.index.union(cal)
    # limit cuenta filas de la union; para no fabricar datos de una serie
    # muerta tambien se corta por antiguedad en dias corridos.
    full = s.reindex(union)
    last_date = pd.Series(np.where(full.notna(), full.index, pd.NaT), index=union)
    last_date = pd.to_datetime(last_date).ffill()
    full = full.ffill()
    age = (union.to_series() - last_date).dt.days
    full[age > max_ffill + 4] = np.nan  # 5 jornadas ~ 9 dias corridos con fin de semana
    out = full.reindex(cal)
    if lag:
        out = out.shift(lag)
    out.name = series.name
    return out


def stale_target_mask(raw: pd.DataFrame) -> pd.Series:
    """Marca jornadas con cotizacion rancia del objetivo (feriados chilenos
    donde Bloomberg repite el precio): retorno exactamente cero y rango
    intradia nulo."""
    px = col(raw, config.TARGET).dropna()
    hi = col(raw, config.TARGET, "PX_HIGH").reindex(px.index)
    lo = col(raw, config.TARGET, "PX_LOW").reindex(px.index)
    zero = px.diff() == 0
    flat = (hi - lo).abs() < 1e-9
    return (zero & flat).rename("stale")


def splice_flow(raw: pd.DataFrame, new: str, old: str) -> pd.Series:
    """Empalma la version nueva (2022+) de la posicion de la banca con la
    antigua (2008-2023). En el traslape de 291 jornadas las dos series tienen
    correlacion de niveles 0.998 (offshore) y 0.984 (local) y razon mediana
    ~1.01: miden lo mismo. Se usa la nueva donde existe."""
    n = col(raw, new).dropna()
    o = col(raw, old).dropna()
    if not n.empty:
        o = o[o.index < n.index.min()]
    return pd.concat([o, n]).sort_index().rename(new)


def macro_point_in_time(raw: pd.DataFrame, ticker: str, cal: pd.DatetimeIndex) -> pd.Series:
    """Ubica cada dato macro en la primera jornada en que ya estaba publicado.

    Donde falta FECHA_RELEASE se asume la fecha de referencia mas un rezago
    conservador (config.MACRO_FALLBACK_LAG_DAYS). Nota: la base entrega el
    valor ya revisado; esto elimina la fuga por rezago, no la fuga por
    revision (se documenta como limitacion)."""
    val = col(raw, ticker).dropna()
    rel = pd.to_datetime(raw[f"{ticker}|FECHA_RELEASE"].reindex(val.index))
    fallback = val.index + pd.Timedelta(days=config.MACRO_FALLBACK_LAG_DAYS[ticker])
    rel = rel.fillna(pd.Series(fallback, index=val.index))
    # Se puede usar al cierre del dia de publicacion: el dato sale en la manana.
    pit = pd.Series(val.values, index=pd.DatetimeIndex(rel.values)).sort_index()
    pit = pit[~pit.index.duplicated(keep="last")]
    out = pit.reindex(pit.index.union(cal)).ffill().reindex(cal)
    return out.rename(ticker)
