"""Senal del dia: direccion esperada, probabilidad y exposicion para manana.

Uso:  python scripts/senal_del_dia.py [ruta_base.xlsx] [--v1]

Por defecto usa la version de produccion (v2). Con --v1, la oficial.

Pensado para correr al cierre de cada jornada. Si se entrega una base nueva,
la usa; si no, la de data/raw. Extiende las predicciones walk-forward (mismo
procedimiento que el backtest), aplica la regla de exposicion y deja:

* reports/senal_del_dia.json  -> ultima senal (para el monitoreo en vivo)
* reports/historial_senales.csv -> todas las senales emitidas
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from usdclp import config, data, features, pipeline, strategy

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    shutil.copy(args[0], config.RAW_XLSX)
    raw = data.load_raw(refresh=True)
else:
    raw = data.load_raw()

df = features.build_dataset(raw)
# Version de produccion: v2 (``--v1`` para la version oficial pre-registrada)
use_v1 = "--v1" in sys.argv
kind = "clf" if use_v1 else pipeline.PRODUCTION_KIND
prm = strategy.FINAL_PARAMS if use_v1 else strategy.V2_PARAMS
pred_path = pipeline.pred_path(kind)
existing = pd.read_parquet(pred_path) if pred_path.exists() else None
pred = pipeline.update_predictions(df, existing, kind=kind)
config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
pred.to_parquet(pred_path)

P = strategy.positions(pred["p_cal"], df["excess_next"], prm, components=pred, vol=df["clp_rv20"])
t = P.index.max()
row = P.loc[t]
pos = int(row["pos"])
if pos == 0 and row.get("vol_state", 0) == 2:
    motivo = "volatilidad extrema (sobre el percentil 95 del ultimo ano): fuera del mercado"
elif pos == 0:
    motivo = "compuerta cerrada: el modelo no ha acertado en las ultimas 250 jornadas" if row["gate"] == 0 \
        else "conviccion bajo el umbral: la ventaja no compensa el costo de operar"
elif abs(pos) == 3:
    motivo = "conviccion en el 10% superior y modelo validado recientemente (IC > %.2f)" % strategy.FINAL_PARAMS.big_ic
else:
    motivo = "conviccion sobre el umbral de abstencion"

next_day = (t + pd.offsets.BDay(1)).date()
signal = {
    "version": "v1 (oficial)" if use_v1 else "v2 (produccion)",
    "fecha_decision": str(t.date()),
    "jornada_objetivo": str(next_day),
    "spot_cierre": float(df.loc[t, "px"]),
    "direccion_esperada": "sube" if row["edge"] > 0 else "baja",
    "probabilidad_alza_modelo": round(float(row["p_model"]), 4),
    "probabilidad_alza_suavizada": round(float(row["p"]), 4),
    "posicion_sugerida": pos,
    "lectura": {0: "fuera del mercado (en pesos)", 1: "comprado 1x", 3: "comprado 3x",
                -1: "vendido 1x", -3: "vendido 3x"}[pos],
    "motivo": motivo,
    "ic_reciente_modelo": None if np.isnan(row["model_ic"]) else round(float(row["model_ic"]), 4),
    "umbral_abstencion": round(float(row["t1"]), 4),
    "umbral_exposicion_maxima": round(float(row["t_big"]), 4),
    # El carry exacto t->t+1 requiere el indice de camara de t+1; al cierre de
    # t se estima con TPM - tasa overnight USD (Act/360).
    "carry_estimado_pb_dia": round(float(df.loc[t, "diff_on"]) / 100 / 360 * 1e4, 3),
}
(config.REPORTS / "senal_del_dia.json").write_text(json.dumps(signal, indent=2, ensure_ascii=False))

hist_path = config.REPORTS / "historial_senales.csv"
hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
hist = pd.concat([hist, pd.DataFrame([{k: v for k, v in signal.items()}])]).drop_duplicates("fecha_decision", keep="last")
hist.to_csv(hist_path, index=False)

print(json.dumps(signal, indent=2, ensure_ascii=False))
