"""Evaluacion fuera de muestra: curvas, metricas, Diebold-Mariano y bootstrap.

Uso:  python scripts/04_evaluar_oos.py

Lee las predicciones walk-forward (scripts/02_predicciones.py) y escribe
tablas en reports/tables y graficos en reports/figures.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from usdclp import backtest as bt
from usdclp import config, features, plots, stats_tests, strategy

config.FIGURES.mkdir(parents=True, exist_ok=True)
config.TABLES.mkdir(parents=True, exist_ok=True)

df = features.build_dataset()
pred = pd.read_parquet(config.DATA_PROCESSED / "pred_wf.parquet")
mask = features.training_mask(df)
ex = df["excess_next"]

P = strategy.positions(pred["p_cal"], ex)
# Ultima jornada con retorno t+1 conocido, sin pasar del cierre oficial del OOS
last_known = min(ex.dropna().index.max(), pd.Timestamp(config.OOS_END))
OOS = slice(config.OOS_START, last_known)
DEV = slice(config.VAL_START, config.DEV_END)

# ---------------------------------------------------------------- referencias
def strategies(Pdf: pd.DataFrame) -> dict[str, pd.Series]:
    idx = Pdf.index
    return {
        "Estrategia": Pdf["pos"],
        "Comprar siempre": pd.Series(1, index=idx),
        "No operar nunca": pd.Series(0, index=idx),
        "Exposicion uno (signo del modelo)": np.sign(Pdf["p_model"] - 0.5).fillna(0).astype(int),
    }


def run_all(period, cost_bps=config.COST_BPS, Pdf=P):
    out = {}
    for name, pos in strategies(Pdf).items():
        out[name] = bt.run(pos.loc[period], ex, cost_bps)
    return out


def metrics_table(runs: dict) -> pd.DataFrame:
    rows = {}
    for name, r in runs.items():
        g = bt.metrics(r, "gross")
        n = bt.metrics(r, "net")
        rows[name] = {
            "Retorno anual bruto %": g["retorno_anual_%"],
            "Retorno anual neto %": n["retorno_anual_%"],
            "Retorno total neto %": n["retorno_total_%"],
            "Volatilidad anual %": n["vol_anual_%"],
            "Sharpe neto": n["sharpe"],
            "Max drawdown %": n["max_drawdown_%"],
            "Acierto dias activos %": n["acierto_dias_activos_%"],
            "Dias activos %": n["dias_activos_%"],
            "Operaciones": n["operaciones"],
            "Exposicion media |x|": n["exposicion_media_abs"],
            "Rotacion anual (x nocional)": n["rotacion_anual"],
            "Costos anuales %": n["costos_anual_%"],
        }
    return pd.DataFrame(rows).T


results = {}
oos_runs = run_all(OOS)
dev_runs = run_all(DEV)
t_oos = metrics_table(oos_runs)
t_dev = metrics_table(dev_runs)
t_oos.round(3).to_csv(config.TABLES / "metricas_oos.csv")
t_dev.round(3).to_csv(config.TABLES / "metricas_desarrollo.csv")
print("\n=== FUERA DE MUESTRA", config.OOS_START, "->", last_known.date(), "===")
print(t_oos.round(2).to_string())
print("\n=== DESARROLLO (walk-forward)", config.VAL_START, "->", config.DEV_END, "===")
print(t_dev.round(2).to_string())

# ---------------------------------------------------------------- por ano
yr = pd.DataFrame({k: v["net"].groupby(v.index.year).sum() * 100 for k, v in run_all(slice(config.VAL_START, last_known)).items()})
yr.round(2).to_csv(config.TABLES / "retorno_neto_por_ano.csv")
print("\nRetorno neto por ano (%)\n", yr.round(1).to_string())

# ---------------------------------------------------------------- calidad de prediccion
V = pred.loc[OOS]
m = mask.reindex(V.index).fillna(False)
y = df["y"].reindex(V.index)
clim = df["y"].where(mask).expanding(250).mean().shift(1).reindex(V.index)  # frecuencia base historica
pq = {
    "AUC": roc_auc_score(y[m], V["p_cal"][m]),
    "Brier modelo": brier_score_loss(y[m], V["p_cal"][m]),
    "Brier p=0.5": brier_score_loss(y[m], np.full(m.sum(), 0.5)),
    "Brier climatologia": brier_score_loss(y[m], clim[m]),
    "LogLoss modelo": log_loss(y[m], V["p_cal"][m]),
    "LogLoss p=0.5": np.log(2),
    "Acierto direccional %": ((V["p_cal"][m] > 0.5) == (y[m] == 1)).mean() * 100,
}
results["calidad_prediccion_oos"] = pq
print("\nCalidad de la prediccion OOS:", {k: round(v, 4) for k, v in pq.items()})

# Diebold-Mariano: (a) perdida de Brier del pronostico probabilistico,
#                  (b) perdida economica = -retorno neto diario de la estrategia.
brier_model = (V["p_cal"] - y) ** 2
dm = {
    "Brier: modelo vs p=0.5 (sin informacion)": stats_tests.diebold_mariano(brier_model[m], ((0.5 - y) ** 2)[m]),
    "Brier: modelo vs climatologia": stats_tests.diebold_mariano(brier_model[m], ((clim - y) ** 2)[m]),
}
for name in ["Comprar siempre", "No operar nunca", "Exposicion uno (signo del modelo)"]:
    dm[f"Retorno neto: estrategia vs {name}"] = stats_tests.diebold_mariano(
        -oos_runs["Estrategia"]["net"], -oos_runs[name]["net"])
t_dm = pd.DataFrame(dm).T
t_dm.to_csv(config.TABLES / "diebold_mariano_oos.csv")
print("\nDiebold-Mariano (OOS, H1: el modelo tiene menor perdida)\n", t_dm.round(4).to_string())

# Bootstrap estacionario sobre el resultado de la estrategia
boot = {}
for name in ["Comprar siempre", "No operar nunca", "Exposicion uno (signo del modelo)"]:
    boot[f"vs {name}"] = stats_tests.bootstrap_test(oos_runs["Estrategia"]["net"], oos_runs[name]["net"])
t_boot = pd.DataFrame(boot).T
t_boot.to_csv(config.TABLES / "bootstrap_oos.csv")
print("\nBootstrap estacionario (OOS, 5000 remuestreos, bloque medio 10)\n", t_boot.to_string())

# Deflated Sharpe: numero de configuraciones probadas en el desarrollo
trials = pd.read_csv(config.TABLES / "grilla_desarrollo.csv") if (config.TABLES / "grilla_desarrollo.csv").exists() else None
n_trials = int(trials.attrs.get("n", len(trials))) if trials is not None else 400
n_trials = max(n_trials, 400)  # incluye ensayos de modelo y variables previos a la grilla
sr_std = float(trials["sharpe"].std()) if trials is not None else 0.3
dsr = stats_tests.deflated_sharpe(oos_runs["Estrategia"]["net"], n_trials, sr_std)
results["deflated_sharpe_oos"] = dsr
print("\nDeflated Sharpe:", {k: round(v, 4) for k, v in dsr.items()})

# ---------------------------------------------------------------- sensibilidad a costos
sens = {}
for c in config.COST_GRID_BPS:
    rr = run_all(OOS, cost_bps=c)
    sens[f"{c:g} pb"] = {k: bt.metrics(v)["retorno_anual_%"] for k, v in rr.items()} | {
        "Sharpe estrategia": bt.metrics(rr["Estrategia"])["sharpe"]}
t_sens = pd.DataFrame(sens).T
t_sens.round(3).to_csv(config.TABLES / "sensibilidad_costos_oos.csv")
print("\nSensibilidad a costos (retorno anual neto %, OOS)\n", t_sens.round(2).to_string())

# ---------------------------------------------------------------- robustez: horario estricto
strict_path = config.DATA_PROCESSED / "pred_wf_strict.parquet"
if strict_path.exists():
    ps = pd.read_parquet(strict_path)
    Ps = strategy.positions(ps["p_cal"], ex)
    rs = {"Estricto (series de NY/Londres rezagadas 1 dia)": bt.run(Ps["pos"].loc[OOS], ex),
          "Estricto - desarrollo": bt.run(Ps["pos"].loc[DEV], ex)}
    t_strict = pd.DataFrame({k: bt.metrics(v) for k, v in rs.items()}).T
    t_strict.round(3).to_csv(config.TABLES / "robustez_horario_estricto.csv")
    print("\nRobustez horario estricto\n", t_strict[["retorno_anual_%", "sharpe", "max_drawdown_%", "dias_activos_%"]].round(2).to_string())

# ---------------------------------------------------------------- robustez: sin capas
abl = {}
for label, prm in {
    "Final": strategy.FINAL_PARAMS,
    "Sin compuerta": strategy.StrategyParams(gate_window=None),
    "Sin suavizado": strategy.StrategyParams(smooth_span=1),
    "Sin exposicion 3": strategy.StrategyParams(big_q=1.0),
    "Sin abstencion (q1=0)": strategy.StrategyParams(q1=0.0),
}.items():
    pp = strategy.positions(pred["p_cal"], ex, prm)
    abl[label] = {**{f"OOS {k}": v for k, v in bt.metrics(bt.run(pp["pos"].loc[OOS], ex)).items() if k in ("retorno_anual_%", "sharpe", "max_drawdown_%")},
                  **{f"DEV {k}": v for k, v in bt.metrics(bt.run(pp["pos"].loc[DEV], ex)).items() if k in ("retorno_anual_%", "sharpe", "max_drawdown_%")}}
t_abl = pd.DataFrame(abl).T
t_abl.round(3).to_csv(config.TABLES / "ablacion_capas.csv")
print("\nAblacion de capas\n", t_abl.round(2).to_string())

# ---------------------------------------------------------------- lectura de la conviccion
R = oos_runs["Estrategia"].join(P[["p", "edge", "model_ic", "gate"]])
R["ret"] = ex.reindex(R.index)
R["abs_ret_bp"] = R["ret"].abs() * 1e4
R["signal_pnl"] = np.sign(R["edge"]) * R["ret"]  # lo que habria hecho la senal con exposicion 1
R["nivel"] = R["pos"].abs().map({0: "0 (abstencion)", 1: "1", 3: "3 (maxima)"})
R["motivo"] = np.where(R["pos"] != 0, "opera", np.where(R["gate"] == 0, "compuerta cerrada", "conviccion baja"))
conv = R.groupby("nivel").agg(
    jornadas=("net", "size"),
    pct_jornadas=("net", lambda s: len(s) / len(R) * 100),
    retorno_neto_total_pct=("net", lambda s: s.sum() * 100),
    acierto_senal_pct=("signal_pnl", lambda s: (s > 0).mean() * 100),
    pnl_senal_x1_bp_dia=("signal_pnl", lambda s: s.mean() * 1e4),
    movimiento_abs_medio_bp=("abs_ret_bp", "mean"),
)
conv.round(2).to_csv(config.TABLES / "conviccion_por_nivel.csv")
abst = R[R["pos"] == 0].groupby("motivo").agg(
    jornadas=("net", "size"), pnl_evitado_senal_x1_pct=("signal_pnl", lambda s: s.sum() * 100),
    acierto_senal_pct=("signal_pnl", lambda s: (s > 0).mean() * 100))
abst.round(2).to_csv(config.TABLES / "abstenciones.csv")
print("\nLectura de la conviccion (OOS)\n", conv.round(2).to_string())
print("\nAbstenciones (OOS)\n", abst.round(2).to_string())

top = R[R["pos"].abs() == 3].copy()
top_tbl = top[["pos", "p", "model_ic", "ret", "net"]].assign(ret=lambda d: d.ret * 1e4, net=lambda d: d.net * 1e4)
top_tbl.columns = ["exposicion", "p_suavizada", "ic_modelo", "variacion_t+1_bp", "resultado_neto_bp"]
top_tbl.round(4).to_csv(config.TABLES / "jornadas_maxima_exposicion.csv")

# ---------------------------------------------------------------- graficos
curves = {k: bt.equity_curve(v) for k, v in oos_runs.items()}
plots.equity_with_exposure(curves, oos_runs["Estrategia"]["pos"],
                           f"Fuera de muestra {config.OOS_START[:4]}-{last_known.year}: neto de carry y {config.COST_BPS:g} pb de costo",
                           config.FIGURES / "curva_oos.png")
full = bt.run(P["pos"].loc[config.VAL_START:last_known], ex)
bh = bt.run(pd.Series(1, index=full.index), ex)
plots.full_history(bt.equity_curve(full), bt.equity_curve(bh), config.OOS_START, config.FIGURES / "curva_historia.png")
plots.reliability(V["p_cal"][m], y[m], config.FIGURES / "calibracion_oos.png", "Calibracion fuera de muestra")
plots.gate_panel(P["model_ic"].loc[config.VAL_START:last_known], P["gate"].loc[config.VAL_START:last_known],
                 strategy.FINAL_PARAMS.big_ic, config.FIGURES / "compuerta.png")
plots.bars(yr["Estrategia"], config.FIGURES / "retorno_por_ano.png", "Retorno neto por ano de la estrategia",
           "% del nocional")
plots.bars(conv["pnl_senal_x1_bp_dia"], config.FIGURES / "conviccion_nivel.png",
           "Ventaja de la senal segun el nivel de exposicion elegido (OOS)", "pb por dia, senal con exposicion 1",
           fmt="{:.2f}")

results.update({
    "periodo_oos": [config.OOS_START, str(last_known.date())],
    "costo_pb": config.COST_BPS,
    "parametros_estrategia": strategy.FINAL_PARAMS.to_dict(),
    "metricas_oos": t_oos.round(4).to_dict(orient="index"),
})
(config.TABLES / "resultados.json").write_text(json.dumps(results, indent=2, default=float, ensure_ascii=False))
print("\nListo: tablas en", config.TABLES, "y figuras en", config.FIGURES)
