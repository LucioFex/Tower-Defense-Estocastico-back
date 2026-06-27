"""
exporter.py — Construye el dict `output.json` según el CONTRATO (RAG/06).

Toma un SimResult + el escenario y produce exactamente el esquema versión 1.0.
"""
from __future__ import annotations

import math

from analytical import analizar
from config import Scenario
from simulation import SimResult, correr

SCHEMA_VERSION = "1.0"


def _series_from_samples(samples: list, c: int) -> dict:
    time = [s["t"] for s in samples]
    queue_len = [s["queue_len"] for s in samples]
    in_system = [s["in_system"] for s in samples]
    tower_temp = [[s["towers"][i]["temp"] for s in samples] for i in range(c)]
    return {"time": time, "queue_len": queue_len, "in_system": in_system,
            "tower_temp": tower_temp}


def build_output(res: SimResult, generated_at: str = "",
                 sweep: list | None = None) -> dict:
    sc = res.scenario
    sim_t = sc.sim_time
    ana = analizar(sc.lam, sc.mu, sc.c, sc.K)

    lambda_eff_sim = res.killed / sim_t if sim_t else 0.0
    avg_wait_q = res.sum_wait_q / res.killed if res.killed else 0.0
    avg_time_sys = res.sum_time_sys / res.killed if res.killed else 0.0
    util = [round(tw.busy_time / sim_t, 4) for tw in res.towers]

    layout = sc.layout
    positions = layout.tower_positions(sc.c)

    out = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "model": "(M/M/c)(FIFO/K/inf)",
            "seed": sc.seed,
            "time_unit": "s",
            "sim_time": sim_t,
            "dt_sample": sc.dt_sample,
            "num_towers": sc.c,
            "queue_capacity": sc.K,
            "generated_at": generated_at,
            "canvas": {"w": layout.canvas_w, "h": layout.canvas_h},
        },
        "params": sc.params_dict(),
        "layout": {
            "spawn": {"x": layout.spawn[0], "y": layout.spawn[1]},
            "base": {"x": layout.base[0], "y": layout.base[1]},
            "path": [
                {"x": layout.spawn[0], "y": layout.spawn[1]},
                {"x": layout.queue_anchor[0], "y": layout.queue_anchor[1]},
                {"x": layout.base[0], "y": layout.base[1]},
            ],
            "queue_anchor": {"x": layout.queue_anchor[0], "y": layout.queue_anchor[1]},
            "towers": [
                {"id": i, "x": positions[i][0], "y": positions[i][1],
                 "range": layout.tower_range}
                for i in range(sc.c)
            ],
        },
        "analytical": {
            "rho": _num(ana.rho), "a": _num(ana.a), "stable": ana.stable,
            "P0": _num(ana.P0), "P_wait": _num(ana.P_wait),
            "L": _num(ana.L), "Lq": _num(ana.Lq),
            "W": _num(ana.W), "Wq": _num(ana.Wq),
            "Pb_finite": _num(ana.Pb_finite), "lambda_eff": _num(ana.lambda_eff),
        },
        "stats": {
            "enemies_spawned": res.spawned,
            "enemies_killed": res.killed,
            "enemies_leaked": res.leaked,
            "leak_rate": round(res.leaked / res.spawned, 4) if res.spawned else 0.0,
            "avg_wait_q": round(avg_wait_q, 4),
            "avg_time_system": round(avg_time_sys, 4),
            "avg_queue_len": round(res.q_area / sim_t, 4),
            "max_queue_len": res.max_queue,
            "avg_in_system": round(res.sys_area / sim_t, 4),
            "rho_sim": round(sum(util) / len(util), 4) if util else 0.0,
            "tower_utilization": util,
            "overheat_events": [tw.overheats for tw in res.towers],
            "base_hp_end": res.base_hp,
        },
        "events": res.events,
        "samples": res.samples,
        "series": _series_from_samples(res.samples, sc.c),
        "sweep": sweep or [],
    }
    return out


def _num(x: float):
    """Serializa infinitos/NaN como None (JSON válido)."""
    if x is None:
        return None
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return round(x, 6) if isinstance(x, float) else x


def run_sweep(base: Scenario, c_values: list[int]) -> list[dict]:
    """Corre el escenario para cada c (mismos lam/mu/K/seed) -> análisis marginal."""
    rows = []
    for c in c_values:
        sc = Scenario(**{**base.__dict__, "c": c})
        sc.layout = base.layout
        res = correr(sc)
        ana = analizar(sc.lam, sc.mu, c, sc.K)
        # M/M/c puro (cola infinita) para Lq/Wq teóricos (null si inestable)
        from analytical import mmc
        ideal = mmc(sc.lam, sc.mu, c)
        rows.append({
            "c": c,
            "stable": ideal.stable,
            "rho": _num(ideal.rho),
            "Lq": _num(ideal.Lq) if ideal.stable else None,
            "Wq": _num(ideal.Wq) if ideal.stable else None,
            "Pb_analytic": _num(ana.Pb_finite),
            "leak_rate_sim": round(res.leaked / res.spawned, 4) if res.spawned else 0.0,
            "Lq_sim": round(res.q_area / sc.sim_time, 4),
            "Wq_sim": round(res.sum_wait_q / res.killed, 4) if res.killed else None,
            "util_sim": round(sum(tw.busy_time for tw in res.towers)
                              / (c * sc.sim_time), 4),
            "overheats": sum(tw.overheats for tw in res.towers),
        })
    return rows
