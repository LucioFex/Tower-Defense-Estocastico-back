"""
config.py — Parámetros del escenario (variables de control del modelo).

Ver RAG/02_variables.md (sección 2.4) para la justificación de cada parámetro.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Layout:
    """Geometría 2D lógica para que Godot reproduzca (no afecta la matemática)."""
    canvas_w: int = 1280
    canvas_h: int = 720
    spawn: tuple[float, float] = (40.0, 360.0)
    base: tuple[float, float] = (1240.0, 360.0)
    queue_anchor: tuple[float, float] = (560.0, 360.0)
    tower_range: float = 200.0

    def tower_positions(self, c: int) -> list[tuple[float, float]]:
        """Distribuye c torres alrededor de la zona de defensa (centro del lienzo)."""
        cx, cy = 700.0, 360.0
        if c == 1:
            return [(cx, cy - 140.0)]
        positions = []
        # mitad arriba, mitad abajo del camino, repartidas en X.
        for i in range(c):
            fila = -1 if i % 2 == 0 else 1
            col = i // 2
            x = cx - 120.0 + col * 130.0
            y = cy + fila * 150.0
            positions.append((x, y))
        return positions


@dataclass
class Scenario:
    # --- Tasas (variables aleatorias) ---
    lam: float = 0.40            # tasa de arribos de enemigos [enemigos/s]
    mu: float = 0.25             # tasa de servicio por torre [kills/s]

    # --- Capacidad (variables de control) ---
    c: int = 3                   # número de torres (servidores)
    K: int = 10                  # capacidad del sistema (cola+servicio); -1 = inf

    # --- Temperatura (variable continua de estado) ---
    T_amb: float = 20.0          # temperatura ambiente
    T_max: float = 100.0         # umbral de sobrecalentamiento
    T_resume: float = 50.0       # umbral de reactivación (histéresis)
    k_heat: float = 2.0          # tasa de calentamiento mientras dispara [°/s]
    k_cool: float = 0.05         # constante de enfriamiento de Newton [1/s]
    # Calibración: con estos valores la temperatura es una perturbación MODERADA
    # (sobrecalentamientos ocasionales en rachas de actividad), no un colapso.
    # Hace que la fuga simulada supere a la analítica M/M/c/K sin volver inestable
    # un sistema que la fórmula declara estable. Ver RAG/05 y RAG/07 (hipótesis H3).

    # --- Daño / base ---
    base_hp: int = 100           # vida inicial de la base
    leak_damage: int = 1         # daño por enemigo que se fuga

    # --- Control de la corrida ---
    sim_time: float = 1200.0     # horizonte de simulación [s]
    dt_sample: float = 0.5       # paso de muestreo para series/samples [s]
    seed: int = 42               # semilla del PRNG (reproducibilidad)

    layout: Layout = field(default_factory=Layout)

    def params_dict(self) -> dict:
        return {
            "lambda": self.lam, "mu": self.mu, "c": self.c, "K": self.K,
            "T_amb": self.T_amb, "T_max": self.T_max, "T_resume": self.T_resume,
            "k_heat": self.k_heat, "k_cool": self.k_cool,
        }


# Escenario por defecto usado por main.py si no se pasa otra cosa.
DEFAULT = Scenario()
