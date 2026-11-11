"""Auditoria de la base: responde con datos las 6 preguntas del diccionario.

Uso:  python scripts/01_auditoria.py

Escribe reports/tables/auditoria_*.csv y reports/figures/auditoria.png.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from usdclp import config, data, features, plots

config.TABLES.mkdir(parents=True, exist_ok=True)
config.FIGURES.mkdir(parents=True, exist_ok=True)
raw = data.load_raw()
cal = data.trading_calendar(raw)
px = data.col(raw, config.TARGET).dropna()
out = {}

# 1. Dolar observado: a que jornada corresponde
obs = data.col(raw, "CLFXDOOB Index").reindex(cal)
d = pd.DataFrame({"do": obs.diff(), "ds": px.diff(), "ds1": px.diff().shift(1)}).dropna()
obs_corr = d.groupby(d.index.year).apply(lambda g: pd.Series({"mismo_dia": g.do.corr(g.ds), "dia_anterior": g.do.corr(g.ds1)}))
obs_corr.round(3).to_csv(config.TABLES / "auditoria_dolar_observado.csv")
out["1_observado"] = (f"Correlacion de variaciones 2000-2026: vs spot del mismo dia {obs_corr.loc[2000:, 'mismo_dia'].mean():.2f}, "
                      f"vs spot del dia anterior {obs_corr.loc[2000:, 'dia_anterior'].mean():.2f}. El observado publicado en t "
                      "refleja la jornada t-1: no sirve para definir el retorno y es redundante como variable.")

# 2. Regimen cambiario
r = np.log(px).diff().dropna()
vol = r.groupby(r.index.year).std() * np.sqrt(252) * 100
zero = r.groupby(r.index.year).apply(lambda v: (v == 0).mean() * 100)
reg = pd.DataFrame({"vol_anual_%": vol, "retornos_cero_%": zero})
reg.round(2).to_csv(config.TABLES / "auditoria_regimen.csv")
out["2_regimen"] = (f"Banda cambiaria hasta sep-1999: 1990-1991 tienen {zero.loc[1990]:.0f}% y {zero.loc[1991]:.0f}% de dias sin "
                    "variacion; desde 2004 la proporcion es ~0-3%. El entrenamiento parte en 2005.")

# 3. Contenido intradia
o = raw[[f"{config.TARGET}|{f}" for f in ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST")]].dropna()
o.columns = ["O", "H", "L", "C"]
flat = (o.H == o.L).groupby(o.index.year).mean()
flat.round(3).to_csv(config.TABLES / "auditoria_intradia.csv")
out["3_intradia"] = (f"Maximo = minimo en {flat.loc[:2003].mean() * 100:.0f}% de los dias 1990-2003 y en "
                     f"{flat.loc[2004:].mean() * 100:.1f}% desde 2004. Las variables de rango intradia solo se usan desde 2004.")

# 4. Quiebres de unidad
chon = data.col(raw, "CHON Index").dropna()
tpm = data.col(raw, "CHOVCHOV Index").dropna()
out["4_quiebres"] = (f"CHON (real, UF) y CHOVCHOV (nominal) el 2001-08-09: {chon.loc['2001-08-09']:.2f}% vs {tpm.loc['2001-08-09']:.2f}%: "
                     "no se empalman. USDBRL antes de 1994 (cruzeiro/cruzeiro real) y WTI negativo (2020-04-20) "
                     "quedan fuera de la ventana o sin log. GTCLP: PX_LAST es tasa, no precio.")

# 5. Huecos
gaps = px.index.to_series().diff().dt.days
big = gaps[gaps > 5]
stale = data.stale_target_mask(raw)
out["5_huecos"] = (f"{len(big)} saltos >5 dias en USDCLP, casi todos entre nov-2011 y oct-2012 (serie con ~120 jornadas faltantes); "
                   f"{int(stale.sum())} cotizaciones rancias (retorno y rango cero; {int(stale.loc['2005':].sum())} desde 2005, feriados chilenos). "
                   "Se excluyen del entrenamiento; el arrastre de datos se corta a las 5 jornadas para no fabricar datos.")
big.to_frame("dias").to_csv(config.TABLES / "auditoria_huecos_usdclp.csv")

# 6. Disponibilidad point-in-time
lags = {}
for t in ["CLIMMOMS Index", "CLIMYOYN Index", "CLINNSMO Index", "CLINNSYO Index", "CHUETOTL Index"]:
    v = data.col(raw, t).dropna()
    rel = pd.to_datetime(raw[f"{t}|FECHA_RELEASE"].reindex(v.index))
    lag = (rel - v.index).dt.days
    lags[t] = {"rezago_mediano_dias": lag.median(), "rezago_max_dias": lag.max(), "sin_fecha_release": int(rel.isna().sum()),
               "rezago_supuesto_si_falta": config.MACRO_FALLBACK_LAG_DAYS[t]}
pd.DataFrame(lags).T.to_csv(config.TABLES / "auditoria_macro_release.csv")
# Flujos: correlacion con el retorno del mismo dia vs rezagado
s = np.log(px).diff()
f = data.col(raw, "CLPBODMB Index").dropna().diff()
j = pd.concat([f, s], axis=1, sort=True).dropna()
j.columns = ["f", "r"]
fl = {k: j.f.corr(j.r.shift(k)) for k in (-1, 0, 1)}
out["6_disponibilidad"] = (f"Macro: rezago mediano de publicacion IMACEC ~34 dias, IPC ~8 dias; se ubican en su FECHA_RELEASE. "
                           f"Flujos BCCh: correlacion con el retorno del mismo dia {fl[0]:.2f} (seria fuga si no se rezaga); "
                           "la extraccion trae el dato hasta un dia antes que el resto, se usan con 2 jornadas de rezago. "
                           "SOFR/Fed funds se publican al dia siguiente: 1 jornada de rezago.")

pd.Series(out, name="hallazgo").to_csv(config.TABLES / "auditoria_resumen.csv")
plots.audit_panel(obs_corr, flat, vol, config.FIGURES / "auditoria.png")

# Quiebre 2025: IC anual de las senales de lead-lag
df = features.build_dataset(raw)
m = features.training_mask(df)
dd = df[m].loc["2005":]
ic = dd.groupby(dd.index.year).apply(lambda g: pd.Series({
    "Pares regionales (r1)": g.peers_r1.corr(g.excess_next, method="spearman"),
    "Rand (r1)": g.usdzar_r1.corr(g.excess_next, method="spearman"),
    "Propio USD/CLP (r1)": g.clp_r1.corr(g.excess_next, method="spearman"),
}))
ic.round(4).to_csv(config.TABLES / "auditoria_quiebre_leadlag.csv")
plots.structural_break(ic, config.FIGURES / "quiebre_2025.png")

for k, v in out.items():
    print(f"[{k}] {v}\n")
