"""Seguimiento fuera de muestra real: jornadas posteriores a la base original.

Uso:  python scripts/06_seguimiento.py [--v1]      (por defecto, la version de produccion v2)

Requiere al menos una descarga en data/raw/actualizaciones/
(actualizacion_hasta_AAAAMMDD.xlsx, generada con la plantilla Bloomberg).

El modelo, los hiperparametros y la regla de exposicion son exactamente los
congelados antes de ver estos datos. Cada decision usa solo la informacion
disponible al cierre de su jornada (mismo procedimiento walk-forward que el
backtest y que el monitoreo en vivo). Con pocas jornadas no hay potencia
estadistica: esto es una verificacion operativa y un primer vistazo, no una
prueba de significancia.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from usdclp import backtest as bt
from usdclp import config, features, pipeline, plots, strategy

use_v1 = "--v1" in sys.argv
kind = "clf" if use_v1 else pipeline.PRODUCTION_KIND
PRM = strategy.FINAL_PARAMS if use_v1 else strategy.V2_PARAMS
VER = "v1" if use_v1 else "v2"
df = features.build_dataset()
official = pd.read_parquet(pipeline.pred_path(kind))
base = official[official.index <= pd.Timestamp(config.FORWARD_START)]
pred = pipeline.update_predictions(df, base, kind=kind)
pred.to_parquet(config.DATA_PROCESSED / f"pred_seguimiento_{VER}.parquet")

# Control: la actualizacion no puede cambiar ninguna prediccion ya emitida.
# (Se excluye la fila del 2026-10-06: con la version anterior del codigo la
# variable de fin de mes de la ultima fila quedaba mal calculada; ver
# features.build_dataset.)
common = base.index[base.index < pd.Timestamp(config.FORWARD_START)].intersection(pred.index)
dmax = (pred.loc[common, "p_cal"] - base.loc[common, "p_cal"]).abs().max()
assert dmax < 1e-9, f"las predicciones historicas cambiaron ({dmax})"

ex = df["excess_next"]
P = strategy.positions(pred["p_cal"], ex, PRM, components=pred, vol=df["clp_rv20"])
start = pd.Timestamp(config.FORWARD_START)
known = ex.dropna().index
fwd = P.loc[(P.index >= start) & (P.index.isin(known))]
if fwd.empty:
    print("No hay jornadas nuevas con resultado conocido todavia.")
    sys.exit(0)


def motivo(r):
    if r.pos == 0 and getattr(r, "vol_state", 0) == 2:
        return "volatilidad extrema"
    if r.pos == 0:
        return "compuerta cerrada" if r.gate == 0 else "conviccion baja"
    return "maxima (x3)" if abs(r.pos) == 3 else "normal (x1)"


refs = {
    "Estrategia": fwd["pos"],
    "Comprar siempre": pd.Series(1, index=fwd.index),
    "No operar nunca": pd.Series(0, index=fwd.index),
    "Exposicion uno (signo del modelo)": np.sign(fwd["p_model"] - 0.5).astype(int),
}
# Para el costo de entrada se parte de la posicion vigente el dia anterior.
prev = P["pos"].shift(1).loc[fwd.index[0]]
runs = {}
for k, pos in refs.items():
    p0 = prev if k == "Estrategia" else (pos.iloc[0] if k != "Exposicion uno (signo del modelo)" else np.sign(P["p_model"].shift(1).loc[fwd.index[0]] - 0.5))
    full = pd.concat([pd.Series([p0], index=[fwd.index[0] - pd.Timedelta(days=1)]), pos])
    runs[k] = bt.run(full, ex.reindex(full.index).fillna(0.0), config.COST_BPS).iloc[1:]

nxt = pd.Series(df.index[1:], index=df.index[:-1])
day = pd.DataFrame({
    "decision": fwd.index.date,
    "jornada": nxt.reindex(fwd.index).dt.date.values,
    "spot_t": df.loc[fwd.index, "px"].values,
    "spot_t+1": df["px"].shift(-1).loc[fwd.index].values,
    "p_suavizada": fwd["p"].round(4).values,
    "posicion": fwd["pos"].values,
    "motivo": [motivo(r) for r in fwd.itertuples()],
    "exceso_t+1_pb": (ex.loc[fwd.index] * 1e4).round(1).values,
    "acierto_direccion": (np.sign(fwd["edge"]) == np.sign(ex.loc[fwd.index])).values,
    "resultado_neto_pb": (runs["Estrategia"]["net"] * 1e4).round(1).values,
})
config.TABLES.mkdir(parents=True, exist_ok=True)
day.to_csv(config.TABLES / f"seguimiento_diario_{VER}.csv", index=False)

summ = pd.DataFrame({k: {
    "jornadas": len(r),
    "resultado_neto_%": r["net"].sum() * 100,
    "resultado_bruto_%": r["gross"].sum() * 100,
    "costos_%": r["cost"].sum() * 100,
    "dias_activos": int((r["pos"] != 0).sum()),
    "acierto_dias_activos_%": (r.loc[r["pos"] != 0, "gross"] > 0).mean() * 100 if (r["pos"] != 0).any() else np.nan,
} for k, r in runs.items()}).T
summ.round(3).to_csv(config.TABLES / f"seguimiento_resumen_{VER}.csv")

pd.set_option("display.width", 200)
print(f"\n=== SEGUIMIENTO {VER} {day['decision'].iloc[0]} -> {day['jornada'].iloc[-1]} ({len(day)} jornadas nunca vistas) ===\n")
print(day.to_string(index=False))
print("\n", summ.round(2).to_string())
act = day[day.posicion != 0]
print(f"\nAcierto direccional de la senal (todas las jornadas): {day.acierto_direccion.mean() * 100:.0f}%"
      f" | en jornadas operadas: {act.acierto_direccion.mean() * 100:.0f}% ({len(act)} jornadas)")

# Grafico
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
a1.axhline(0, color=plots.GRID, lw=0.8, zorder=0)
for (k, r), c in zip(runs.items(), [plots.BLUE, plots.ORANGE, plots.AQUA, plots.MUTED]):
    cum = r["net"].cumsum() * 100
    x = nxt.reindex(r.index).values
    a1.plot(x, cum.values, color=c, lw=2.2 if c == plots.BLUE else 1.4, marker="o", ms=3, label=k)
    a1.annotate(f"{cum.iloc[-1]:+.1f}%", (x[-1], cum.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                color=plots.INK2, fontsize=9, va="center")
a1.set_ylabel("Resultado neto acumulado, %")
a1.set_title(f"Seguimiento real {VER}: {len(day)} jornadas posteriores a la base (modelo congelado)", loc="left", pad=26)
a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
a2.bar(nxt.reindex(fwd.index).values, fwd["pos"].values, color=plots.BLUE, width=0.7)
a2.set_yticks([-3, -1, 0, 1, 3])
a2.set_ylabel("Exposicion")
a2.axhline(0, color=plots.MUTED, lw=0.8)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(config.FIGURES / f"seguimiento_{VER}.png", dpi=150)
plt.close(fig)

last = P.index.max()
r = P.loc[last]
print(f"\nUltima decision disponible ({last.date()}): posicion {int(r.pos)}, p suavizada {r.p:.4f}, "
      f"IC modelo {r.model_ic:.3f} -> {motivo(r)}")
