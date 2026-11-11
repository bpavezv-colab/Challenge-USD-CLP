"""Evaluacion UNICA de la version 2 elegida en scripts/07_v2_seleccion.py.

Uso:  python scripts/08_v2_evaluar.py

Declaracion: la v2 se diseno despues de ver el fuera de muestra 2023-2026 de
la v1 (las ideas vienen de la revision de proyectos y literatura, y la
eleccion entre candidatas se hizo solo con 2013-2022). Por eso su cifra
2023-2026 no es tan limpia como la de la v1; la prueba limpia de la v2 es el
seguimiento posterior y el monitoreo en vivo.

Incluye:
  * v2 vs v1 vs referencias: desarrollo, fuera de muestra y seguimiento.
  * Diebold-Mariano y bootstrap estacionario de la v2.
  * Alfa y beta contra comprar siempre (Newey-West): cuanto es timing.
  * Control nulo: el mismo pipeline con etiquetas barajadas.
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
from sklearn.metrics import roc_auc_score

from usdclp import backtest as bt
from usdclp import config, entrega, features, plots, stats_tests, strategy

df = features.build_dataset()
ex = df["excess_next"]
mask = features.training_mask(df)
choice = json.loads((config.TABLES / "v2_eleccion.json").read_text())
VOL = {"ninguna": (None, None), "sin_x3_p80": (0.8, None), "sin_x3_p80_fuera_p95": (0.8, 0.95)}
cq, oq = VOL[choice["volatilidad"]]
agree = str(choice["acuerdo"]).lower() == "true"
PRM_V2 = strategy.StrategyParams(agree_filter=agree, vol_cap_q=cq, vol_off_q=oq)
kind = choice["modelo"]
pred_v2 = pd.read_parquet(config.DATA_PROCESSED / ("pred_wf.parquet" if kind == "clf" else f"pred_wf_{kind}.parquet"))
pred_v1 = pd.read_parquet(config.DATA_PROCESSED / "pred_wf.parquet")

P2 = strategy.positions(pred_v2["p_cal"], ex, PRM_V2, components=pred_v2, vol=df["clp_rv20"])
P1 = strategy.positions(pred_v1["p_cal"], ex)

known = ex.dropna().index.max()
PER = {
    "Desarrollo 2013-2022": slice(config.VAL_START, config.DEV_END),
    "Fuera de muestra 2023 - 5-oct-2026": slice(config.OOS_START, config.OOS_END),
    "Seguimiento 6-oct - ultimo dato": slice(config.FORWARD_START, known),
}


def runs(sl):
    idx = P1.loc[sl].index
    return {
        "v2": bt.run(P2["pos"].loc[sl], ex),
        "v1 (oficial)": bt.run(P1["pos"].loc[sl], ex),
        "Comprar siempre": bt.run(pd.Series(1, index=idx), ex),
        "No operar nunca": bt.run(pd.Series(0, index=idx), ex),
        "Exposicion uno": bt.run(np.sign(P1["p_model"].loc[sl] - 0.5).fillna(0).astype(int), ex),
    }


rows = []
for per, sl in PER.items():
    for name, r in runs(sl).items():
        m = bt.metrics(r)
        rows.append({"periodo": per, "estrategia": name, "jornadas": m["dias"],
                     "retorno_neto_%": m["retorno_total_%"], "retorno_anual_neto_%": m["retorno_anual_%"],
                     "sharpe_neto": m["sharpe"], "max_dd_%": m["max_drawdown_%"],
                     "dias_activos_%": m["dias_activos_%"], "acierto_dias_activos_%": m["acierto_dias_activos_%"],
                     "operaciones": m["operaciones"]})
T = pd.DataFrame(rows)
T.round(3).to_csv(config.TABLES / "v2_comparacion.csv", index=False)
pd.set_option("display.width", 200)
print("Configuracion v2:", kind, "| acuerdo:", agree, "| volatilidad:", choice["volatilidad"])
print(T.round(2).to_string(index=False))

# Pruebas de significancia de la v2 en 2023-2026
OOS = PER["Fuera de muestra 2023 - 5-oct-2026"]
R = runs(OOS)
tests = {}
for ref in ["Comprar siempre", "No operar nunca", "Exposicion uno", "v1 (oficial)"]:
    b = stats_tests.bootstrap_test(R["v2"]["net"], R[ref]["net"])
    d = stats_tests.diebold_mariano(-R["v2"]["net"], -R[ref]["net"])
    tests[f"v2 vs {ref}"] = {"dif_retorno_anual_%": b["dif_retorno_anual_%"],
                             "p_bootstrap": b["p_valor_dif_retorno"], "DM": d["DM"], "p_DM": d["p_valor_una_cola"]}
ab = {k: stats_tests.alpha_beta(R[k]["net"], R["Comprar siempre"]["gross"]) for k in ["v2", "v1 (oficial)"]}
T2 = pd.DataFrame(tests).T
T3 = pd.DataFrame(ab).T
T2.round(4).to_csv(config.TABLES / "v2_tests_oos.csv")
T3.round(4).to_csv(config.TABLES / "alfa_beta_oos.csv")
print("\nPruebas v2 (2023-2026)\n", T2.round(3).to_string())
print("\nAlfa / beta contra comprar siempre (Newey-West)\n", T3.round(3).to_string())

# Control nulo
null_path = config.DATA_PROCESSED / "pred_wf_null.parquet"
if null_path.exists():
    pn = pd.read_parquet(null_path)
    out = {}
    for per, sl in list(PER.items())[:2]:
        V = pn.loc[sl]
        m = mask.reindex(V.index).fillna(False)
        yv = df["y"].reindex(V.index)
        Pn = strategy.positions(pn["p_cal"], ex)
        rn = bt.run(Pn["pos"].loc[sl], ex)
        out[per] = {"AUC": roc_auc_score(yv[m], V["p_cal"][m]),
                    "IC_spearman": pd.Series(V["p_cal"] - 0.5).corr(ex.reindex(V.index), method="spearman"),
                    "retorno_anual_neto_%": bt.metrics(rn)["retorno_anual_%"],
                    "sharpe_neto": bt.metrics(rn)["sharpe"]}
    TN = pd.DataFrame(out).T
    TN.round(4).to_csv(config.TABLES / "control_nulo.csv")
    print("\nControl nulo (etiquetas barajadas)\n", TN.round(3).to_string())

# Grafico: historia completa v2 vs v1 vs comprar siempre
full = slice(config.VAL_START, known)
Rf = runs(full)
fig_curves = {"v2": bt.equity_curve(Rf["v2"]), "v1 (oficial)": bt.equity_curve(Rf["v1 (oficial)"]),
              "Comprar siempre": bt.equity_curve(Rf["Comprar siempre"])}
plots.equity_with_exposure(fig_curves, Rf["v2"]["pos"],
                           "v2 vs v1, 2013 - hoy (neto de carry y 4 pb). Desarrollo hasta 2022, luego fuera de muestra",
                           config.FIGURES / "v2_vs_v1.png", colors=[plots.BLUE, plots.VIOLET, plots.ORANGE])
# Posiciones dia a dia (para que el monitor del laboratorio replique)
entrega.export_positions(df, P2.loc[config.VAL_START:], config.REPORTS / "posiciones_v2.csv")
entrega.export_positions(df, P1.loc[config.VAL_START:], config.REPORTS / "posiciones_v1.csv")
print("\nListo. Posiciones diarias en reports/posiciones_v2.csv y posiciones_v1.csv")
