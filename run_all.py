"""Ejecuta el proyecto completo desde la base cruda. Funciona en Windows, macOS y Linux.

Uso:
    python run_all.py            # todo: auditoria, modelos, evaluaciones, senal (~15 min)
    python run_all.py --rapido   # solo lo necesario para la estrategia de produccion
                                 # y sus estadisticas (~5 min)

Requisito: la base del laboratorio en data/raw/base_usdclp.xlsx.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "raw" / "base_usdclp.xlsx"

COMPLETO = [
    ["scripts/01_auditoria.py"],
    ["scripts/02_predicciones.py"],                      # v1 (clasificador)
    ["scripts/02_predicciones.py", "--kind", "reg"],     # v2 (produccion)
    ["scripts/02_predicciones.py", "--strict"],          # robustez horaria
    ["scripts/02_predicciones.py", "--kind", "combo"],   # candidata v2
    ["scripts/02_predicciones.py", "--kind", "null"],    # control nulo
    ["scripts/03_grilla_desarrollo.py"],
    ["scripts/04_evaluar_oos.py"],
    ["scripts/05_analisis_posthoc.py"],
    ["scripts/07_v2_seleccion.py"],
    ["scripts/08_v2_evaluar.py"],
    ["scripts/06_seguimiento.py"],
    ["scripts/06_seguimiento.py", "--v1"],
    ["scripts/senal_del_dia.py"],
]
RAPIDO = [
    ["scripts/02_predicciones.py"],
    ["scripts/02_predicciones.py", "--kind", "reg"],
    ["scripts/04_evaluar_oos.py"],
    ["scripts/08_v2_evaluar.py"],
    ["scripts/06_seguimiento.py"],
    ["scripts/senal_del_dia.py"],
]


def main() -> int:
    if not BASE.exists():
        print(f"Falta la base del laboratorio: copiala en {BASE}")
        return 1
    steps = RAPIDO if "--rapido" in sys.argv else COMPLETO
    t0 = time.time()
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] python {' '.join(step)}", flush=True)
        t = time.time()
        res = subprocess.run([sys.executable, *step], cwd=ROOT)
        if res.returncode != 0:
            print(f"ERROR en {' '.join(step)} (codigo {res.returncode})")
            return res.returncode
        print(f"    ok ({time.time() - t:.0f} s)")
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min. Resultados en reports/, senal en reports/senal_del_dia.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
