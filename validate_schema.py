"""
validate_schema.py — Auto-auditoría del contrato output.json (RAG/06).

Verifica los 8 invariantes del contrato. Lo usa main.py ANTES de exportar
(falla la build si algo no cumple) y puede correrse standalone sobre un archivo:

    python validate_schema.py output.json
"""
from __future__ import annotations

import json
import sys

REQUIRED_TOP = ["meta", "params", "layout", "analytical", "stats",
                "events", "samples", "series", "sweep"]
EVENT_TYPES = {"spawn", "enqueue", "start_service", "kill", "leak",
               "overheat", "cooldown_done"}
TOWER_STATES = {"idle", "busy", "cooldown"}


def validate(data: dict) -> tuple[bool, list[str]]:
    err: list[str] = []

    # estructura mínima
    for k in REQUIRED_TOP:
        if k not in data:
            err.append(f"falta la clave de nivel superior '{k}'")
    if err:
        return False, err

    meta, layout, series, samples = (data["meta"], data["layout"],
                                     data["series"], data["samples"])
    c = meta["num_towers"]

    # Inv 1: len(towers) == num_towers == params.c
    if len(layout["towers"]) != c or data["params"]["c"] != c:
        err.append("Inv1: num_towers != len(layout.towers) != params.c")

    # Inv 2: events ordenado por t no decreciente
    ts = [e["t"] for e in data["events"]]
    if any(b < a for a, b in zip(ts, ts[1:])):
        err.append("Inv2: events no está ordenado por t ascendente")

    # Inv 3: todo kill/leak tuvo spawn previo; tipos válidos
    vivos = set()
    spawned = set()
    for e in data["events"]:
        if e["type"] not in EVENT_TYPES:
            err.append(f"Inv3: tipo de evento desconocido '{e['type']}'")
            continue
        if e["type"] == "spawn":
            spawned.add(e["enemy_id"]); vivos.add(e["enemy_id"])
        elif e["type"] in ("kill", "leak"):
            if e["enemy_id"] not in spawned:
                err.append(f"Inv3: {e['type']} de enemy_id={e['enemy_id']} sin spawn")
            vivos.discard(e["enemy_id"])

    # Inv 4: largos de series == samples
    n = len(samples)
    if not (len(series["time"]) == len(series["queue_len"])
            == len(series["in_system"]) == n):
        err.append("Inv4: series.time/queue_len/in_system no coinciden con samples")

    # Inv 5: len(series.tower_temp) == num_towers
    if len(series["tower_temp"]) != c:
        err.append("Inv5: series.tower_temp no tiene num_towers filas")

    # Inv 6: balance de enemigos
    st = data["stats"]
    en_sistema = len(vivos)
    if st["enemies_spawned"] != st["enemies_killed"] + st["enemies_leaked"] + en_sistema:
        err.append(f"Inv6: spawn({st['enemies_spawned']}) != kill+leak+en_sistema "
                   f"({st['enemies_killed']}+{st['enemies_leaked']}+{en_sistema})")

    # Inv 7: temp >= 0 y estados válidos
    for s in samples:
        for tw in s["towers"]:
            if tw["temp"] < 0:
                err.append(f"Inv7: temp negativa en t={s['t']}"); break
            if tw["state"] not in TOWER_STATES:
                err.append(f"Inv7: estado inválido '{tw['state']}'"); break

    # Inv 8: si K = -1 (infinito) no debe haber leaks
    if meta["queue_capacity"] == -1:
        if any(e["type"] == "leak" for e in data["events"]):
            err.append("Inv8: hay eventos 'leak' con capacidad infinita (K=-1)")

    return (len(err) == 0), err


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "output.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ok, err = validate(data)
    if ok:
        print(f"[OK] '{path}' cumple el contrato v{data['meta']['schema_version']}.")
    else:
        print(f"[X] '{path}' NO cumple el contrato:")
        for e in err:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
