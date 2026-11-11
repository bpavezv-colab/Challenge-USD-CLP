"""Predicciones walk-forward 2008-hoy (reentrenamiento mensual).

Uso:  python scripts/02_predicciones.py [--strict]

Cada mes se reentrena con todo lo anterior (menos un embargo de 5 jornadas) y
se predice ese mes. Ninguna prediccion usa datos posteriores a su fecha, de
modo que estas probabilidades son fuera de muestra en todo el periodo.
``--strict`` repite el ejercicio rezagando un dia las series que cierran
despues de Santiago (prueba de robustez del horario).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from usdclp import config, features, pipeline

strict = "--strict" in sys.argv
kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "clf"
df = features.build_dataset(strict_timing=strict)
config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
if not strict:
    df.to_parquet(config.DATA_PROCESSED / "dataset.parquet")
t = time.time()
pred = pipeline.predict_walk_forward(df, end=str(df.index.max().date()), kind=kind)
suffix = ("_strict" if strict else "") + ("" if kind == "clf" else f"_{kind}")
out = config.DATA_PROCESSED / f"pred_wf{suffix}.parquet"
pred.to_parquet(out)
print(f"{len(pred)} predicciones, {pred['refit'].nunique()} reentrenamientos, {time.time() - t:.0f}s -> {out}")
