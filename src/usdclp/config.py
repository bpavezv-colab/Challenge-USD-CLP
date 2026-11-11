"""Parametros centrales del proyecto.

Todo lo que define la evaluacion (fechas de corte, costos, grilla de exposicion)
vive aqui, para que ningun numero quede escondido en un script.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models"

RAW_XLSX = DATA_RAW / "base_usdclp.xlsx"
RAW_PARQUET = DATA_RAW / "datos.parquet"
# Descargas posteriores con la plantilla Bloomberg (actualizacion_hasta_AAAAMMDD.xlsx)
UPDATES_DIR = DATA_RAW / "actualizaciones"

TARGET = "USDCLP Curncy"

# --- Ventanas -------------------------------------------------------------
# Chile abandona la banda cambiaria el 2-sep-1999 (libre flotacion). Ademas el
# rango intradia de USDCLP solo tiene contenido desde 2004 y varias series
# explicativas (BBDXY, swaps CLP) nacen 2003-2005. Por eso el entrenamiento
# parte en 2005.
TRAIN_START = "2005-01-01"
# Tramo de desarrollo: todo lo que el modelo puede ver para elegir
# hiperparametros y umbrales.
DEV_END = "2022-12-31"
# Validacion walk-forward dentro del desarrollo (umbrales y seleccion de modelo).
VAL_START = "2013-01-01"
# Fuera de muestra: tramo final que el modelo nunca vio.
OOS_START = "2023-01-01"
# Ultima jornada del fuera de muestra oficial (la base del laboratorio llega al
# 2026-10-06; el ultimo retorno conocido es el de la decision del 2026-10-05).
# Lo posterior es seguimiento: scripts/06_seguimiento.py.
OOS_END = "2026-10-05"
FORWARD_START = "2026-10-06"

# Embargo entre entrenamiento y prueba (dias habiles). El objetivo mira 1 dia
# adelante, de modo que 5 dias es holgado (Lopez de Prado, cap. 7).
EMBARGO_DAYS = 5

# --- Estrategia -----------------------------------------------------------
EXPOSURES = (-3, -1, 0, 1, 3)

# Costo de transaccion por unidad de nocional transada, en puntos base.
# Spread tipico interbancario USD/CLP ~0.3-0.5 CLP sobre ~950 => 3-5 pb ida y
# vuelta. Usamos 4 pb por unidad de cambio de exposicion como caso base y
# reportamos sensibilidad. Reemplazar por el parametro del codigo oficial
# del laboratorio cuando corresponda.
COST_BPS = 4.0
COST_GRID_BPS = (0.0, 2.0, 4.0, 6.0, 10.0)

# Rezagos de publicacion (dias habiles) aplicados a las variables explicativas.
PUBLICATION_LAG = {
    # Publicadas el dia siguiente en Nueva York.
    "SOFRRATE Index": 1,
    "FEDL01 Index": 1,
    # Posiciones de la banca publicadas por el BCCh con rezago. La extraccion
    # del 2026-10-07 trae el dato hasta el 2026-10-05 mientras el resto llega
    # al 2026-10-06: rezago >= 1 dia habil. Usamos 2 por prudencia.
    "CLPBODMB Index": 2,
    "CLPBLDMB Index": 2,
    "CLPBSMB Index": 2,
    "CLPBODM Index": 2,
    "CLPBLDM Index": 2,
}

# Rezago conservador (dias corridos) para macro sin FECHA_RELEASE: el maximo
# observado en las filas que si la traen, mas holgura.
MACRO_FALLBACK_LAG_DAYS = {
    "CLIMMOMS Index": 45,
    "CLIMYOYN Index": 50,
    "CLINNSMO Index": 12,
    "CLINNSYO Index": 12,
    "CHUETOTL Index": 40,
}

RANDOM_STATE = 42
