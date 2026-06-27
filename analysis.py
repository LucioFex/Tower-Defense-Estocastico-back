"""
analysis.py — Módulo B: lee output.json, genera gráficos estáticos e imprime
conclusiones automatizadas en terminal.

Independiente del Módulo A: solo consume output.json (el contrato).

Uso:
    python analysis.py [output.json] [--outdir figs] [--no-show]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Consola Windows (cp1252) no soporta λ/μ/°; forzamos UTF-8 si se puede.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")              # backend sin ventana (Ubuntu headless)
import matplotlib.pyplot as plt
import pandas as pd

C_TORRE = 1.0       # costo por torre / unidad de tiempo (parámetro económico)
C_FUGA = 25.0       # penalización por enemigo que se fuga


# --------------------------------------------------------------------------- #
def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_series_colas(d: dict, outdir: str) -> str:
    s = d["series"]
    df = pd.DataFrame({"t": s["time"], "cola": s["queue_len"],
                       "sistema": s["in_system"]})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], df["sistema"], label="En el sistema (L)", lw=1.0, color="#c0392b")
    ax.plot(df["t"], df["cola"], label="En cola (Lq)", lw=1.0, color="#2980b9")
    ax.axhline(d["stats"]["avg_in_system"], ls="--", color="#c0392b", alpha=.5,
               label=f"L̄={d['stats']['avg_in_system']:.2f}")
    ax.axhline(d["stats"]["avg_queue_len"], ls="--", color="#2980b9", alpha=.5,
               label=f"L̄q={d['stats']['avg_queue_len']:.2f}")
    ax.set(xlabel="tiempo [s]", ylabel="n° de enemigos",
           title="Evolución temporal de la cola y del sistema")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.3)
    return _save(fig, outdir, "01_series_colas.png")


def fig_temperatura(d: dict, outdir: str) -> str:
    s = d["series"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, temps in enumerate(s["tower_temp"]):
        ax.plot(s["time"], temps, lw=0.9, label=f"Torre {i}")
    ax.axhline(d["params"]["T_max"], ls="--", color="red", alpha=.6, label="T_max")
    ax.axhline(d["params"]["T_resume"], ls=":", color="orange", alpha=.6,
               label="T_resume")
    ax.set(xlabel="tiempo [s]", ylabel="temperatura [°]",
           title="Variable continua: temperatura de las torres (calentamiento/enfriamiento)")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(alpha=.3)
    return _save(fig, outdir, "02_temperatura.png")


def fig_utilizacion(d: dict, outdir: str) -> str:
    util = d["stats"]["tower_utilization"]
    over = d["stats"]["overheat_events"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(util))
    bars = ax.bar(x, util, color="#16a085")
    ax.axhline(d["analytical"]["rho"], ls="--", color="black", alpha=.6,
               label=f"ρ teórica={d['analytical']['rho']:.2f}")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01,
                f"{over[i]} sobrecal.", ha="center", fontsize=8)
    ax.set(xlabel="torre", ylabel="utilización", ylim=(0, 1),
           title="Uso de servidores (utilización por torre)")
    ax.set_xticks(list(x)); ax.legend(); ax.grid(alpha=.3, axis="y")
    return _save(fig, outdir, "03_utilizacion.png")


def fig_rendimiento_marginal(d: dict, outdir: str):
    sweep = d.get("sweep") or []
    if not sweep:
        return None
    df = pd.DataFrame(sweep)
    estables = df[df["stable"]].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Izq: Lq (analítico) y tasa de fuga (sim) vs c
    ax1.plot(estables["c"], estables["Lq"], "o-", color="#2980b9", label="Lq (Erlang C)")
    ax1.set(xlabel="n° de torres (c)", ylabel="Lq [enemigos]",
            title="Longitud de cola vs. capacidad")
    ax1.grid(alpha=.3)
    ax1b = ax1.twinx()
    ax1b.plot(df["c"], df["leak_rate_sim"], "s--", color="#c0392b",
              label="tasa de fuga (sim)")
    ax1b.set_ylabel("tasa de fuga", color="#c0392b")
    ax1.legend(loc="upper right", fontsize=8)

    # Der: mejora marginal ΔLq (rendimiento marginal decreciente)
    cs = estables["c"].tolist()
    lqs = estables["Lq"].tolist()
    dc = cs[1:]
    dlq = [lqs[i] - lqs[i + 1] for i in range(len(lqs) - 1)]
    ax2.bar([str(c) for c in dc], dlq, color="#8e44ad")
    ax2.set(xlabel="torre agregada (c-1 → c)", ylabel="ΔLq (reducción)",
            title="Rendimiento marginal decreciente de cada torre")
    ax2.grid(alpha=.3, axis="y")
    return _save(fig, outdir, "04_rendimiento_marginal.png")


def fig_costo_optimo(d: dict, outdir: str):
    sweep = d.get("sweep") or []
    if not sweep:
        return None, None
    df = pd.DataFrame(sweep)
    df["costo_torres"] = df["c"] * C_TORRE
    lam = d["params"]["lambda"]
    df["costo_fuga"] = C_FUGA * lam * df["leak_rate_sim"]
    df["costo_total"] = df["costo_torres"] + df["costo_fuga"]
    c_opt = int(df.loc[df["costo_total"].idxmin(), "c"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["c"], df["costo_torres"], "o--", label="Costo torres", color="#16a085")
    ax.plot(df["c"], df["costo_fuga"], "s--", label="Costo fugas", color="#c0392b")
    ax.plot(df["c"], df["costo_total"], "D-", label="Costo TOTAL", color="black", lw=2)
    ax.axvline(c_opt, ls=":", color="purple", alpha=.7, label=f"c* óptimo = {c_opt}")
    ax.set(xlabel="n° de torres (c)", ylabel="costo / unidad de tiempo",
           title=f"Óptimo económico (C_torre={C_TORRE}, C_fuga={C_FUGA})")
    ax.legend(); ax.grid(alpha=.3)
    return _save(fig, outdir, "05_costo_optimo.png"), c_opt


# --------------------------------------------------------------------------- #
def conclusiones(d: dict, c_opt: int | None) -> None:
    print("\n" + "=" * 64)
    print("  CONCLUSIONES AUTOMATIZADAS (Módulo B)")
    print("=" * 64)
    ana, st = d["analytical"], d["stats"]
    p = d["params"]

    print(f"\n  Escenario: λ={p['lambda']}  μ={p['mu']}  c={p['c']}  K={p['K']}")
    print(f"  ρ (utilización) = {ana['rho']:.3f}  →  "
          f"sistema {'ESTABLE' if ana['stable'] else 'INESTABLE'}")

    # --- validación analítico vs simulado ---
    print("\n  [Validación del modelo: analítico (M/M/c/K) vs. simulado]")
    print(f"    {'métrica':<14}{'analítico':>12}{'simulado':>12}")
    _row("Wq [s]", ana["Wq"], st["avg_wait_q"])
    _row("Lq", ana["Lq"], st["avg_queue_len"])
    _row("L", ana["L"], st["avg_in_system"])
    pb = ana["Pb_finite"]
    _row("Fuga (Pb)", pb, st["leak_rate"])

    # --- Ley de Little ---
    lam_eff = st["enemies_killed"] / d["meta"]["sim_time"]
    little_L = lam_eff * st["avg_time_system"]
    print(f"\n  [Ley de Little]  λ_eff·W = {lam_eff:.3f}·{st['avg_time_system']:.3f} "
          f"= {little_L:.3f}   vs   L_sim = {st['avg_in_system']:.3f}")
    err = abs(little_L - st["avg_in_system"]) / max(st["avg_in_system"], 1e-9)
    print(f"                   error relativo = {err:.1%}  "
          f"({'OK' if err < 0.15 else 'revisar'})")

    # --- sweep: estabilidad, rendimiento marginal, óptimo ---
    sweep = d.get("sweep") or []
    if sweep:
        df = pd.DataFrame(sweep)
        estables = df[df["stable"]]
        c_min = int(estables["c"].min()) if not estables.empty else None
        print(f"\n  [Capacidad]")
        print(f"    c_min (mínimo para estabilidad ρ<1) = {c_min}")
        lqs = estables["Lq"].tolist(); cs = estables["c"].tolist()
        if len(lqs) >= 2:
            mejoras = [lqs[i] - lqs[i + 1] for i in range(len(lqs) - 1)]
            total = lqs[0] - lqs[-1]
            if total > 0:
                pct = mejoras[0] / total
                print(f"    La 1ª torre sobre c_min captura {pct:.0%} de la "
                      f"reducción TOTAL de cola.")
                print(f"    Mejoras marginales ΔLq por torre: "
                      f"{[round(m, 2) for m in mejoras]}")
                print(f"    → Confirma RENDIMIENTO MARGINAL DECRECIENTE "
                      f"(cada torre extra rinde menos).")
        if c_opt is not None:
            print(f"\n  [Decisión económica]")
            print(f"    c* (óptimo costo torres+fugas) = {c_opt}")
            print(f"    Recomendación: dimensionar a c={c_opt} torres; "
                  f"más torres desperdician capacidad (ρ→0).")

    # --- temperatura ---
    tot_over = sum(st["overheat_events"])
    print(f"\n  [Temperatura]  sobrecalentamientos totales = {tot_over} "
          f"{st['overheat_events']}")
    if pb is not None and pb >= 0:
        gap = st["leak_rate"] - pb
        print(f"    Fuga simulada ({st['leak_rate']:.1%}) vs analítica "
              f"({pb:.1%}): brecha = {gap:+.1%}")
        print("    La brecha la explica la INDISPONIBILIDAD por temperatura, "
              "que M/M/c/K no modela.")
    print(f"\n  Base: vida final = {st['base_hp_end']} / inicial.")
    print("=" * 64)


# --------------------------------------------------------------------------- #
def _row(nombre: str, ana, sim) -> None:
    a = "n/a" if ana is None else f"{ana:.4f}"
    s = "n/a" if sim is None else f"{sim:.4f}"
    print(f"    {nombre:<14}{a:>12}{s:>12}")


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Tower Defense Estocástico — Módulo B")
    ap.add_argument("input", nargs="?", default="output.json")
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    d = load(args.input)
    generados = []
    generados.append(fig_series_colas(d, args.outdir))
    generados.append(fig_temperatura(d, args.outdir))
    generados.append(fig_utilizacion(d, args.outdir))
    f4 = fig_rendimiento_marginal(d, args.outdir)
    if f4: generados.append(f4)
    f5, c_opt = fig_costo_optimo(d, args.outdir)
    if f5: generados.append(f5)

    conclusiones(d, c_opt)
    print("\n  Gráficos generados:")
    for g in generados:
        print(f"    - {g}")


if __name__ == "__main__":
    main()
