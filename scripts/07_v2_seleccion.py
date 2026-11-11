"""Version 2: candidatas inspiradas en la revision de proyectos y literatura,
SELECCIONADAS SOLO CON EL DESARROLLO 2013-2022.

Uso:  python scripts/07_v2_seleccion.py

Requiere las predicciones walk-forward de cada modelo base:
    python scripts/02_predicciones.py                 (clf, v1)
    python scripts/02_predicciones.py --kind reg
    python scripts/02_predicciones.py --kind combo
    python scripts/02_predicciones.py --kind null

Candidatas (todas con la misma regla de exposicion v1 de base):
  * Modelo base: clasificador (v1) | regresion sobre retorno normalizado por
    volatilidad | combinacion de ambos.
  * Filtro de acuerdo entre modelos lineales y de arboles: si | no.
  * Gestion por volatilidad: ninguna | sin x3 sobre el p80 de vol |
    sin x3 sobre el p80 y fuera sobre el p95.

Se elige la de mayor Sharpe neto de desarrollo; el tramo 2023+ no se mira
aqui. La evaluacion de la elegida va en scripts/08_v2_evaluar.py.
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import pandas as pd

from usdclp import backtest as bt
from usdclp import config, features, strategy

df = features.build_dataset()
ex = df["excess_next"]
DEV = slice(config.VAL_START, config.DEV_END)
preds = {
    "clf": pd.read_parquet(config.DATA_PROCESSED / "pred_wf.parquet"),
    "reg": pd.read_parquet(config.DATA_PROCESSED / "pred_wf_reg.parquet"),
    "combo": pd.read_parquet(config.DATA_PROCESSED / "pred_wf_combo.parquet"),
}
VOL = {"ninguna": (None, None), "sin_x3_p80": (0.8, None), "sin_x3_p80_fuera_p95": (0.8, 0.95)}

rows = []
for kind, agree, vname in itertools.product(preds, [False, True], VOL):
    pr = preds[kind]
    cq, oq = VOL[vname]
    prm = strategy.StrategyParams(agree_filter=agree, vol_cap_q=cq, vol_off_q=oq)
    P = strategy.positions(pr["p_cal"], ex, prm, components=pr, vol=df["clp_rv20"])
    r = bt.run(P["pos"].loc[DEV], ex)
    m = bt.metrics(r)
    yr = r["net"].groupby(r.index.year).sum()
    rows.append({"modelo": kind, "acuerdo": agree, "volatilidad": vname, "sharpe": m["sharpe"],
                 "retorno_anual_%": m["retorno_anual_%"], "max_drawdown_%": m["max_drawdown_%"],
                 "dias_activos_%": m["dias_activos_%"], "anos_positivos": int((yr > 0).sum()),
                 "peor_ano_%": yr.min() * 100})
R = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
config.TABLES.mkdir(parents=True, exist_ok=True)
R.round(4).to_csv(config.TABLES / "v2_grilla_desarrollo.csv", index=False)
print(R.round(3).to_string(index=False))
best = R.iloc[0].to_dict()
(config.TABLES / "v2_eleccion.json").write_text(json.dumps(best, indent=2, default=str, ensure_ascii=False))
print("\nElegida (solo desarrollo):", best)
