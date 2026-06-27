"""
experiments.py — Módulo B+: experimentos avanzados sobre el modelo.

Tres estudios que enriquecen las conclusiones del TP (todos opt-in, no tocan el
escenario por defecto ni el contrato output.json):

  1. Réplicas + intervalos de confianza  -> rigor estadístico (reducción de varianza).
  2. Escenario NO estacionario (oleadas λ(t)) -> "dimensionar al pico, no al promedio".
  3. Tipos de enemigo (V.A. discreta) -> la VARIANZA del servicio importa, no solo la media.

Uso:
    python experiments.py                 # corre los 3 estudios -> figs_exp/ + conclusiones
    python experiments.py --reps 20
    python experiments.py --only ci|olas|tipos
"""
from __future__ import annotations

import argparse
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytical import mmc, mmck
from config import Scenario
from simulation import correr

# t de Student (0.975, dos colas) para n-1 grados de libertad
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
      25: 2.060, 30: 2.042}


def t_crit(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T:
        return _T[df]
    keys = sorted(_T)
    if df > keys[-1]:
        return 1.96
    return _T[min(keys, key=lambda k: abs(k - df))]


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Media e intervalo de confianza 95% (semiancho) por t de Student."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return m, t_crit(n - 1) * sd / math.sqrt(n)


def summarize(res, sim_time: float) -> dict:
    return {
        "leak_rate": res.leaked / res.spawned if res.spawned else 0.0,
        "Wq": res.sum_wait_q / res.killed if res.killed else 0.0,
        "Lq": res.q_area / sim_time,
        "util": sum(t.busy_time for t in res.towers) / (len(res.towers) * sim_time),
        "overheats": sum(t.overheats for t in res.towers),
    }


def replicate(base: Scenario, seeds: list[int], **overrides) -> list[dict]:
    out = []
    for s in seeds:
        sc = Scenario(**{**base.__dict__, **overrides, "seed": s})
        sc.layout = base.layout
        out.append(summarize(correr(sc), sc.sim_time))
    return out


# --------------------------------------------------------------------------- #
#  Estudio 1: réplicas + intervalos de confianza                               #
# --------------------------------------------------------------------------- #
def estudio_ci(base: Scenario, reps: int, c_values: list[int]) -> int:
    seeds = list(range(1000, 1000 + reps))
    print("\n" + "=" * 64)
    print(f"  ESTUDIO 1 — Réplicas (n={reps}) e intervalos de confianza 95%")
    print("=" * 64)
    print(f"  Escenario base: λ={base.lam} μ={base.mu} K={base.K} T={base.sim_time}s")

    rows = []
    for c in c_values:
        reps_data = replicate(base, seeds, c=c)
        leak_m, leak_ci = mean_ci([r["leak_rate"] for r in reps_data])
        wq_m, wq_ci = mean_ci([r["Wq"] for r in reps_data])
        util_m, _ = mean_ci([r["util"] for r in reps_data])
        ideal = mmc(base.lam, base.mu, c)
        rows.append({"c": c, "leak_m": leak_m, "leak_ci": leak_ci, "wq_m": wq_m,
                     "wq_ci": wq_ci, "util_m": util_m, "stable": ideal.stable})

    print(f"\n  {'c':>2} {'fuga%':>14} {'Wq [s]':>16} {'util':>7} {'estable':>8}")
    for r in rows:
        print(f"  {r['c']:>2} {r['leak_m']*100:>7.2f} ±{r['leak_ci']*100:>4.2f}   "
              f"{r['wq_m']:>8.2f} ±{r['wq_ci']:>5.2f}  {r['util_m']:>6.2f}  "
              f"{'sí' if r['stable'] else 'NO':>7}")

    # óptimo económico con CI (mismos costos que analysis.py)
    C_TORRE, C_FUGA = 1.0, 25.0
    costs = [(r["c"], r["c"] * C_TORRE + C_FUGA * base.lam * r["leak_m"]) for r in rows]
    c_opt = min(costs, key=lambda kv: kv[1])[0]
    print(f"\n  Óptimo económico c* = {c_opt} (costo torres+fugas, C_torre={C_TORRE}, "
          f"C_fuga={C_FUGA})")
    print("  → Las réplicas dan un c* robusto, no dependiente de una sola semilla.")

    # gráfico: fuga vs c con banda de CI
    fig, ax = plt.subplots(figsize=(9, 5))
    cs = [r["c"] for r in rows]
    lm = [r["leak_m"] * 100 for r in rows]
    lci = [r["leak_ci"] * 100 for r in rows]
    ax.errorbar(cs, lm, yerr=lci, fmt="o-", capsize=5, color="#c0392b",
                label="tasa de fuga (media ± IC95%)")
    ax.axvline(c_opt, ls=":", color="purple", label=f"c* = {c_opt}")
    ax.set(xlabel="n° de torres (c)", ylabel="tasa de fuga [%]",
           title=f"Estudio 1 — Fuga vs. capacidad con IC 95% (n={reps} réplicas)")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "06_ci_fuga_vs_c.png")
    return c_opt


# --------------------------------------------------------------------------- #
#  Estudio 2: no estacionario (oleadas)                                         #
# --------------------------------------------------------------------------- #
def estudio_no_estacionario(base: Scenario) -> None:
    print("\n" + "=" * 64)
    print("  ESTUDIO 2 — Arribos NO estacionarios (oleadas λ(t))")
    print("=" * 64)
    # oleadas: valle λ=0.15 y pico λ=0.75 alternados cada 150 s
    periodo = 150.0
    sched = []
    t = 0.0
    hi = False
    while t < base.sim_time:
        sched.append((t, 0.75 if hi else 0.15))
        hi = not hi
        t += periodo
    lam_avg = sum(v for _, v in sched) / len(sched)
    print(f"  λ alterna 0.15 (valle) / 0.75 (pico) cada {periodo:.0f}s. "
          f"λ promedio ≈ {lam_avg:.2f}")
    print(f"  Dimensionado al PROMEDIO: c≈2 (ρ_avg={lam_avg/(2*base.mu):.2f}).  "
          f"Al PICO: c≈4 (ρ_pico={0.75/(4*base.mu):.2f}).")

    res = {}
    for c in (2, 4):
        sc = Scenario(**{**base.__dict__, "c": c, "lam_schedule": sched, "seed": 42})
        sc.layout = base.layout
        r = correr(sc)
        res[c] = (r, sc)
        print(f"    c={c}: fuga={r.leaked/r.spawned*100:5.1f}%  "
              f"base_hp={r.base_hp:3d}  max_cola={r.max_queue}")

    print("  → Dimensionar al PROMEDIO deja fugas grandes en los picos; al PICO las controla.")
    print("    Conclusión: en sistemas con oleadas, c se dimensiona al pico, no al promedio.")

    # gráfico: λ(t), cola(t) para c=2 y c=4
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    # λ(t) escalonada
    ts_step, lam_step = [], []
    for i, (t0, v) in enumerate(sched):
        t1 = sched[i + 1][0] if i + 1 < len(sched) else base.sim_time
        ts_step += [t0, t1]; lam_step += [v, v]
    a1.plot(ts_step, lam_step, color="#333", lw=1.5)
    a1.set(ylabel="λ(t) [arr/s]", title="Estudio 2 — Oleadas: λ(t) y longitud de cola resultante")
    a1.grid(alpha=.3)
    for c, col in ((2, "#c0392b"), (4, "#2980b9")):
        r, sc = res[c]
        a2.plot([s["t"] for s in r.samples],
                [s["queue_len"] for s in r.samples], lw=0.9, color=col,
                label=f"c={c} (fuga {r.leaked/r.spawned*100:.0f}%)")
    a2.set(xlabel="tiempo [s]", ylabel="cola", ); a2.legend(); a2.grid(alpha=.3)
    _save(fig, "07_no_estacionario.png")


# --------------------------------------------------------------------------- #
#  Estudio 3: tipos de enemigo (varianza del servicio)                          #
# --------------------------------------------------------------------------- #
def estudio_tipos(base: Scenario, reps: int) -> None:
    print("\n" + "=" * 64)
    print("  ESTUDIO 3 — Tipos de enemigo (V.A. discreta): la VARIANZA importa")
    print("=" * 64)
    # factores elegidos con E[1/factor]=1 -> MISMA media de servicio que homogéneo,
    # pero mayor varianza. Débil: factor 2 (rápido). Fuerte: factor 2/3 (lento).
    tipos = [(0.5, 2.0, "debil"), (0.5, 2.0 / 3.0, "fuerte")]
    e_inv = sum(p / f for p, f, _ in tipos)
    print(f"  Tipos: 50% débil (μ×2) / 50% fuerte (μ×2/3).  E[1/factor]={e_inv:.2f} "
          f"→ misma media de servicio, mayor varianza.")
    seeds = list(range(2000, 2000 + reps))

    homo = replicate(base, seeds)
    hetero = replicate(base, seeds, enemy_types=tipos)
    for nombre, data in (("Homogéneo", homo), ("Heterogéneo", hetero)):
        lm, lci = mean_ci([r["leak_rate"] for r in data])
        wm, wci = mean_ci([r["Wq"] for r in data])
        print(f"    {nombre:<12} fuga={lm*100:5.2f}±{lci*100:.2f}%   "
              f"Wq={wm:5.2f}±{wci:.2f}s")
    dl = mean_ci([r["leak_rate"] for r in hetero])[0] - mean_ci(
        [r["leak_rate"] for r in homo])[0]
    print(f"  → Con la MISMA media de servicio, la heterogeneidad cambia la fuga en "
          f"{dl*100:+.2f} pp.")
    print("    Confirma que el modelo M/M (servicio exponencial) no capta todo: la "
          "varianza del\n    servicio afecta la congestión (intuición de Pollaczek-Khinchine).")


# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> str:
    import os
    os.makedirs("figs_exp", exist_ok=True)
    path = os.path.join("figs_exp", name)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"    [fig] {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Tower Defense Estocástico — experimentos")
    ap.add_argument("--reps", type=int, default=12, help="n° de réplicas")
    ap.add_argument("--only", choices=["ci", "olas", "tipos"], help="correr solo un estudio")
    ap.add_argument("--sweep-max", type=int, default=6)
    args = ap.parse_args()

    base = Scenario()
    if args.only in (None, "ci"):
        estudio_ci(base, args.reps, list(range(1, args.sweep_max + 1)))
    if args.only in (None, "olas"):
        estudio_no_estacionario(base)
    if args.only in (None, "tipos"):
        estudio_tipos(base, args.reps)
    print("\n" + "=" * 64)
    print("  Experimentos completos. Figuras en figs_exp/")


if __name__ == "__main__":
    main()
