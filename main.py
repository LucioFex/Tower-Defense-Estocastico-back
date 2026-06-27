"""
main.py — Módulo A: ejecuta la simulación y exporta output.json.

Uso:
    python main.py                          # escenario por defecto + sweep c=1..6
    python main.py --c 4 --lam 0.5 --mu 0.25 --sim 1200 --seed 7
    python main.py --no-sweep --out salida.json

El output.json resultante es consumido por:
  - analysis.py (Módulo B, gráficos y conclusiones)
  - el frontend Godot (reproducción visual)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import Scenario
from exporter import build_output, run_sweep
from simulation import correr
from validate_schema import validate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tower Defense Estocástico — Módulo A")
    p.add_argument("--lam", type=float, help="tasa de arribos (enemigos/s)")
    p.add_argument("--mu", type=float, help="tasa de servicio por torre (kills/s)")
    p.add_argument("--c", type=int, help="número de torres")
    p.add_argument("--K", type=int, help="capacidad del sistema (-1 = infinito)")
    p.add_argument("--sim", type=float, help="tiempo de simulación (s)")
    p.add_argument("--seed", type=int, help="semilla del PRNG")
    p.add_argument("--out", default="output.json", help="archivo de salida")
    p.add_argument("--no-sweep", action="store_true", help="no correr el barrido de c")
    p.add_argument("--sweep-max", type=int, default=6, help="c máximo del barrido")
    return p.parse_args()


def build_scenario(args: argparse.Namespace) -> Scenario:
    sc = Scenario()
    if args.lam is not None: sc.lam = args.lam
    if args.mu is not None: sc.mu = args.mu
    if args.c is not None: sc.c = args.c
    if args.K is not None: sc.K = args.K
    if args.sim is not None: sc.sim_time = args.sim
    if args.seed is not None: sc.seed = args.seed
    return sc


def main() -> None:
    args = parse_args()
    sc = build_scenario(args)

    print("=" * 64)
    print("  TOWER DEFENSE ESTOCÁSTICO — Módulo A (Simulación SimPy)")
    print("=" * 64)
    print(f"  Modelo (M/M/c)(FIFO/K/inf):  c={sc.c}  K={sc.K}")
    print(f"  lambda={sc.lam}  mu={sc.mu}  rho={sc.lam/(sc.c*sc.mu):.3f}"
          f"  sim_time={sc.sim_time}s  seed={sc.seed}")
    print("-" * 64)

    res = correr(sc)
    print(f"  Enemigos: spawn={res.spawned}  kill={res.killed}  "
          f"fuga={res.leaked}  base_hp={res.base_hp}")

    sweep = []
    if not args.no_sweep:
        c_values = list(range(1, args.sweep_max + 1))
        print(f"  Corriendo barrido de torres c={c_values} ...")
        sweep = run_sweep(sc, c_values)

    generated_at = datetime.now().isoformat(timespec="seconds")
    output = build_output(res, generated_at=generated_at, sweep=sweep)

    # --- AUTO-AUDITORÍA: el esquema debe cumplir el contrato antes de exportar ---
    ok, errores = validate(output)
    if not ok:
        print("\n[X] El output NO cumple el contrato (RAG/06). Errores:")
        for e in errores:
            print(f"    - {e}")
        raise SystemExit(1)
    print("  [OK] Auto-auditoría del esquema superada (contrato v"
          f"{output['meta']['schema_version']}).")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)
    n_ev = len(output["events"])
    n_sa = len(output["samples"])
    print(f"  [OK] Exportado '{args.out}'  ({n_ev} eventos, {n_sa} muestras)")
    print("=" * 64)
    print("  Siguiente: python analysis.py   (gráficos + conclusiones)")


if __name__ == "__main__":
    main()
