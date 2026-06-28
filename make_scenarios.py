"""
make_scenarios.py — genera un SET de escenarios output.json para el SELECTOR del front.

Cada escenario es una corrida válida y reproducible (no rompe la regla de oro ni el
supuesto de Poisson): solo cambian variables de control del modelo (lam, c, disciplina).
El front los carga en caliente al hacer click en los iconos del selector.

Se usa dt_sample=1.0 (en vez de 0.5) para achicar los archivos a la mitad; el front
interpola igual. El output.json "canónico" de 1200s/0.5 (escenario Normal / c=3 FIFO)
NO se toca: el selector lo reutiliza para esos dos botones.

Uso:  python make_scenarios.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from config import Scenario
from exporter import build_output, run_sweep
from simulation import correr
from validate_schema import validate

OUT_DIR = os.path.join("..", "Tower-Defense-Estocastico-front", "scenarios")

# Mezcla de enemigos para la disciplina de cola (V.A. discreta):
#   (prob, factor_mu, nombre).  factor_mu>1 = débil (se mata rápido); <1 = fuerte (lento).
ENEMY_MIX = [(0.6, 1.5, "goblin"), (0.4, 0.6, "orco")]

# (id, overrides del Scenario). dt_sample=1.0 se aplica a todos en make().
SCEN = [
    # --- Carga (varía lambda; c=3 FIFO) ---
    ("carga_tranquilo", dict(lam=0.20)),
    ("carga_saturado", dict(lam=0.70)),
    # --- Dimensionado (varía c; lambda=0.40 FIFO). c=3 == Normal (reusa output.json) ---
    ("c1", dict(c=1)),
    ("c2", dict(c=2)),
    ("c4", dict(c=4)),
    ("c5", dict(c=5)),
    ("c6", dict(c=6)),
    # --- Disciplina de cola (mezcla de enemigos; lambda=0.40 c=3) ---
    ("cola_fifo", dict(enemy_types=ENEMY_MIX, priority=False)),
    ("cola_prioridad", dict(enemy_types=ENEMY_MIX, priority=True)),
]


def make(scen_id: str, overrides: dict) -> None:
    sc = Scenario(dt_sample=1.0, **overrides)
    res = correr(sc)
    sweep = run_sweep(sc, list(range(1, 7)))
    out = build_output(res, generated_at=datetime.now().isoformat(timespec="seconds"),
                       sweep=sweep)
    ok, errs = validate(out)
    if not ok:
        raise SystemExit(f"[X] {scen_id} NO cumple el contrato: {errs}")
    path = os.path.join(OUT_DIR, scen_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    kb = os.path.getsize(path) // 1024
    print(f"  [OK] {scen_id:16s} lam={sc.lam:<4} c={sc.c} prio={int(sc.priority)}  "
          f"{kb} KB  ({len(out['events'])} ev, {len(out['samples'])} muestras)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Generando escenarios para el SELECTOR del frontend")
    print("=" * 60)
    for sid, ov in SCEN:
        make(sid, ov)
    print("-" * 60)
    print(f"  Listo. Archivos en {OUT_DIR}")
    print("  (el escenario Normal / c=3 reusa el output.json canónico)")


if __name__ == "__main__":
    main()
