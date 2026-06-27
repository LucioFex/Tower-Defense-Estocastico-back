"""
simulation.py — Módulo A: simulación por eventos discretos con SimPy.

Modelo (M/M/c)(FIFO/K/inf) + dinámica continua de temperatura por torre.
Ver RAG/03_modelo_de_colas.md y RAG/05_recursos_y_temperatura.md.

Idea clave: la cola FIFO única se modela con un `simpy.Store`. Cada torre es un
proceso que hace `yield queue.get()` (FIFO). Una torre en COOLDOWN simplemente NO
pide enemigos -> capacidad efectiva reducida (server breakdown). Esto rompe el
supuesto markoviano y es justamente lo que hace necesaria la simulación.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import simpy
from simpy.resources.store import PriorityItem

from config import Scenario
from prng import ExponentialStream


# --------------------------------------------------------------------------- #
#  Torre: máquina de estados con temperatura continua (integración por tramos) #
# --------------------------------------------------------------------------- #
class Tower:
    IDLE, BUSY, COOLDOWN = "idle", "busy", "cooldown"

    def __init__(self, tid: int, sc: Scenario):
        self.id = tid
        self.sc = sc
        self.temp = sc.T_amb
        self.state = Tower.IDLE
        self._last_t = 0.0
        self._heating = False          # True mientras dispara (BUSY)
        # métricas
        self.busy_time = 0.0
        self.overheats = 0

    def temp_at(self, now: float) -> float:
        """Temperatura en el instante `now` integrando el tramo actual (cerrado)."""
        dt = now - self._last_t
        if dt <= 0:
            return self.temp
        if self._heating:
            return self.temp + self.sc.k_heat * dt          # rampa lineal
        # Ley de enfriamiento de Newton (decaimiento exponencial)
        Ta = self.sc.T_amb
        return Ta + (self.temp - Ta) * math.exp(-self.sc.k_cool * dt)

    def _commit(self, now: float) -> None:
        self.temp = temp_clamp(self.temp_at(now))
        self._last_t = now

    def go_busy(self, now: float) -> None:
        self._commit(now)
        self._heating = True
        self.state = Tower.BUSY

    def go_idle(self, now: float) -> None:
        self._commit(now)
        self._heating = False
        self.state = Tower.IDLE

    def go_cooldown(self, now: float) -> None:
        self._commit(now)
        self._heating = False
        self.state = Tower.COOLDOWN

    def cooldown_duration(self, now: float) -> float:
        """Tiempo (Newton) para enfriar de la temp actual hasta T_resume."""
        self._commit(now)
        Ta, Tr = self.sc.T_amb, self.sc.T_resume
        if self.temp <= Tr:
            return 0.0
        return (1.0 / self.sc.k_cool) * math.log((self.temp - Ta) / (Tr - Ta))


def temp_clamp(t: float) -> float:
    return t if t > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Resultado de una corrida                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    scenario: Scenario
    events: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    # contadores
    spawned: int = 0
    killed: int = 0
    leaked: int = 0
    base_hp: int = 0
    # acumuladores de integrales temporales
    q_area: float = 0.0
    sys_area: float = 0.0
    max_queue: int = 0
    sum_wait_q: float = 0.0
    sum_time_sys: float = 0.0
    towers: list = field(default_factory=list)
    # espera en cola desagregada por tipo de enemigo (para el estudio de prioridades)
    wait_by_type: dict = field(default_factory=dict)
    n_by_type: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Simulador                                                                    #
# --------------------------------------------------------------------------- #
class TowerDefenseSim:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.env = simpy.Environment()
        # Cola: FIFO (Store) o por prioridad no-preemptiva (PriorityStore).
        # Con prioridad, el "fuerte" (menor factor_mu) se atiende antes al liberarse una torre.
        self.queue = simpy.PriorityStore(self.env) if sc.priority else simpy.Store(self.env)
        self.towers = [Tower(i, sc) for i in range(sc.c)]
        # Streams INDEPENDIENTES por fuente de aleatoriedad (números aleatorios comunes):
        # así activar tipos de enemigo NO perturba la secuencia de arribos, y comparar
        # escenarios (homogéneo vs heterogéneo, c distinto) aísla el efecto estudiado.
        self.stream_arr = ExponentialStream(sc.seed)            # arribos
        self.stream_srv = ExponentialStream(sc.seed + 10_007)   # servicio
        self.stream_type = ExponentialStream(sc.seed + 20_011)  # tipo de enemigo
        self.serving = 0                              # enemigos en servicio
        self.res = SimResult(scenario=sc)
        self.res.base_hp = sc.base_hp
        self._last_int_t = 0.0
        self._next_eid = 1

    # ---- helpers ---------------------------------------------------------- #
    def _ev(self, t: float, etype: str, **kw) -> None:
        ev = {"t": round(t, 4), "type": etype}
        ev.update(kw)
        self.res.events.append(ev)

    def _in_system(self) -> int:
        return len(self.queue.items) + self.serving

    def _integrate(self) -> None:
        """Acumula áreas (queue_len, in_system) hasta env.now antes de un cambio."""
        now = self.env.now
        dt = now - self._last_int_t
        if dt > 0:
            q = len(self.queue.items)
            self.res.q_area += q * dt
            self.res.sys_area += self._in_system() * dt
        self._last_int_t = now

    # ---- extensiones opt-in ---------------------------------------------- #
    def _current_lam(self) -> float:
        """Tasa de arribos vigente (no estacionario si hay lam_schedule)."""
        if not self.sc.lam_schedule:
            return self.sc.lam
        now = self.env.now
        lam = self.sc.lam_schedule[0][1]
        for t0, val in self.sc.lam_schedule:
            if now >= t0:
                lam = val
            else:
                break
        return lam

    def _pick_type(self) -> tuple[float, str]:
        """Sortea el tipo del enemigo EN EL SPAWN: devuelve (factor_mu, nombre).

        Homogéneo (sin enemy_types): no consume aleatoriedad -> default reproducible.
        """
        if not self.sc.enemy_types:
            return 1.0, "uniforme"
        r = self.stream_type.uniform()
        acc = 0.0
        for prob, factor, nombre in self.sc.enemy_types:
            acc += prob
            if r <= acc:
                return factor, nombre
        prob, factor, nombre = self.sc.enemy_types[-1]
        return factor, nombre

    # ---- procesos --------------------------------------------------------- #
    def arrivals(self):
        """Proceso generador de arribos: Exp(lambda), estacionario o por tramos."""
        while True:
            yield self.env.timeout(self.stream_arr.sample(self._current_lam()))
            now = self.env.now
            eid = self._next_eid
            self._next_eid += 1
            self.res.spawned += 1
            self._ev(now, "spawn", enemy_id=eid)

            # ¿Hay lugar? K finito -> posible fuga (bloqueo).
            if self.sc.K > 0 and self._in_system() >= self.sc.K:
                self.res.leaked += 1
                self.res.base_hp = max(0, self.res.base_hp - self.sc.leak_damage)
                self._ev(now, "leak", enemy_id=eid, base_hp=self.res.base_hp)
                continue

            self._integrate()
            factor, tipo = self._pick_type()          # el tipo se fija al aparecer
            enemy = {"id": eid, "arrival": now, "mu_factor": factor, "tipo": tipo}
            if self.sc.priority:
                # prioridad = factor_mu (menor = más fuerte = antes); +eid desempata por FIFO
                prio = factor + eid * 1e-6
                self.queue.put(PriorityItem(prio, enemy))
            else:
                self.queue.put(enemy)
            self._ev(now, "enqueue", enemy_id=eid, queue_len=len(self.queue.items))
            self.res.max_queue = max(self.res.max_queue, len(self.queue.items))

    def tower_proc(self, tower: Tower):
        """Proceso de una torre: toma el siguiente enemigo (FIFO o prioridad) y lo atiende."""
        while True:
            item = yield self.queue.get()             # bloquea si la cola está vacía
            enemy = item.item if self.sc.priority else item
            now = self.env.now
            self._integrate()
            self.serving += 1
            wait = now - enemy["arrival"]
            self.res.sum_wait_q += wait
            tipo = enemy.get("tipo", "uniforme")
            self.res.wait_by_type[tipo] = self.res.wait_by_type.get(tipo, 0.0) + wait
            self.res.n_by_type[tipo] = self.res.n_by_type.get(tipo, 0) + 1
            tower.go_busy(now)
            self._ev(now, "start_service", enemy_id=enemy["id"], tower_id=tower.id)

            ts = self.stream_srv.sample(self.sc.mu * enemy["mu_factor"])
            busy_start = now
            yield self.env.timeout(ts)

            now = self.env.now
            self._integrate()
            tower.busy_time += now - busy_start
            tower.go_idle(now)                        # commit -> temp ya calentada
            self.serving -= 1
            self.res.killed += 1
            self.res.sum_time_sys += now - enemy["arrival"]
            self._ev(now, "kill", enemy_id=enemy["id"], tower_id=tower.id)

            # ¿Se sobrecalentó? Termina el kill y entra en cooldown forzado.
            if tower.temp >= self.sc.T_max:
                tower.overheats += 1
                tower.go_cooldown(now)
                self._ev(now, "overheat", tower_id=tower.id, temp=round(tower.temp, 2))
                cd = tower.cooldown_duration(now)
                yield self.env.timeout(cd)
                now = self.env.now
                tower.temp = self.sc.T_resume
                tower._last_t = now
                tower.go_idle(now)
                self._ev(now, "cooldown_done", tower_id=tower.id,
                         temp=round(tower.temp, 2))

    def monitor(self):
        """Toma una foto del estado en una grilla uniforme dt_sample."""
        while True:
            now = self.env.now
            self.res.samples.append({
                "t": round(now, 4),
                "queue_len": len(self.queue.items),
                "in_system": self._in_system(),
                "towers": [
                    {"id": tw.id, "temp": round(tw.temp_at(now), 2),
                     "state": tw.state, "busy": tw.state == Tower.BUSY}
                    for tw in self.towers
                ],
            })
            yield self.env.timeout(self.sc.dt_sample)

    # ---- ejecución -------------------------------------------------------- #
    def run(self) -> SimResult:
        self.env.process(self.arrivals())
        for tw in self.towers:
            self.env.process(self.tower_proc(tw))
        self.env.process(self.monitor())
        self.env.run(until=self.sc.sim_time)

        # cierre de integrales
        self._last_int_t = min(self._last_int_t, self.sc.sim_time)
        self.env._now = self.sc.sim_time
        self._integrate()
        self.res.towers = self.towers
        return self.res


def correr(sc: Scenario) -> SimResult:
    """Ejecuta una corrida y devuelve el resultado crudo."""
    return TowerDefenseSim(sc).run()


if __name__ == "__main__":
    from config import DEFAULT
    r = correr(DEFAULT)
    sim_t = DEFAULT.sim_time
    print(f"spawned={r.spawned} killed={r.killed} leaked={r.leaked} "
          f"base_hp={r.base_hp}")
    print(f"Lq_sim={r.q_area / sim_t:.3f}  L_sim={r.sys_area / sim_t:.3f}  "
          f"Wq_sim={r.sum_wait_q / max(1, r.killed):.3f}")
    for tw in r.towers:
        print(f"  torre {tw.id}: util={tw.busy_time / sim_t:.3f} "
              f"overheats={tw.overheats}")
