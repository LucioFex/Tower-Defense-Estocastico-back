"""
analytical.py — Fórmulas cerradas de Teoría de Colas (Unidad III).

Modelos:
  - M/M/1            (caso c = 1, fórmulas del apunte de la materia)
  - M/M/c            (Erlang C, cola infinita)  -> cota teórica de control
  - M/M/c/K          (capacidad finita)         -> probabilidad de bloqueo = "fuga"

Todas las funciones reciben tasas: lam (arribos), mu (servicio por servidor), c, K.
Ver RAG/03_modelo_de_colas.md para las derivaciones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class QueueMetrics:
    rho: float
    a: float
    stable: bool
    P0: float
    P_wait: float
    L: float
    Lq: float
    W: float
    Wq: float
    Pb_finite: float          # prob. de bloqueo M/M/c/K (None-> -1 si K infinito)
    lambda_eff: float

    def as_dict(self) -> dict:
        return asdict(self)


def _erlang_c(c: int, a: float, rho: float) -> tuple[float, float]:
    """Devuelve (P0, P_wait) para M/M/c con a = lam/mu, rho = a/c < 1."""
    suma = sum(a ** n / math.factorial(n) for n in range(c))
    ultimo = (a ** c / math.factorial(c)) * (1.0 / (1.0 - rho))
    P0 = 1.0 / (suma + ultimo)
    P_wait = ultimo * P0
    return P0, P_wait


def mmc(lam: float, mu: float, c: int) -> QueueMetrics:
    """M/M/c con cola infinita (Erlang C). Reduce a M/M/1 si c = 1."""
    a = lam / mu
    rho = a / c
    stable = rho < 1.0

    if not stable:
        # Sistema inestable: la cola diverge; no hay métricas finitas.
        return QueueMetrics(rho=rho, a=a, stable=False, P0=0.0, P_wait=1.0,
                            L=math.inf, Lq=math.inf, W=math.inf, Wq=math.inf,
                            Pb_finite=-1.0, lambda_eff=lam)

    P0, P_wait = _erlang_c(c, a, rho)
    Lq = P_wait * rho / (1.0 - rho)
    Wq = Lq / lam
    W = Wq + 1.0 / mu
    L = Lq + a
    return QueueMetrics(rho=rho, a=a, stable=True, P0=P0, P_wait=P_wait,
                        L=L, Lq=Lq, W=W, Wq=Wq, Pb_finite=-1.0, lambda_eff=lam)


def mmck(lam: float, mu: float, c: int, K: int) -> QueueMetrics:
    """M/M/c/K (capacidad finita K = cola + servicio). Calcula prob. de bloqueo.

    Soporta rho >= 1 (con K finito el sistema siempre es estable: la cola no
    puede crecer más allá de K).
    """
    if K < c:
        raise ValueError("K debe ser >= c")
    a = lam / mu
    rho = a / c

    # p(n) sin normalizar, tomando p(0) = 1.
    p = [0.0] * (K + 1)
    p[0] = 1.0
    for n in range(1, K + 1):
        if n <= c:
            p[n] = p[n - 1] * a / n
        else:
            p[n] = p[n - 1] * a / c
    total = sum(p)
    p = [x / total for x in p]

    P0 = p[0]
    Pb = p[K]                      # bloqueo: sistema lleno -> fuga
    lambda_eff = lam * (1.0 - Pb)
    L = sum(n * p[n] for n in range(K + 1))
    W = L / lambda_eff if lambda_eff > 0 else math.inf
    Lq = L - lambda_eff / mu
    Wq = Lq / lambda_eff if lambda_eff > 0 else math.inf
    P_wait = sum(p[n] for n in range(c, K))   # esperan (servidores llenos, hay lugar)

    return QueueMetrics(rho=rho, a=a, stable=True, P0=P0, P_wait=P_wait,
                        L=L, Lq=Lq, W=W, Wq=Wq, Pb_finite=Pb,
                        lambda_eff=lambda_eff)


def analizar(lam: float, mu: float, c: int, K: int = -1) -> QueueMetrics:
    """Punto de entrada: usa M/M/c/K si K es finito (>0), si no M/M/c."""
    if K is not None and K > 0:
        return mmck(lam, mu, c, K)
    return mmc(lam, mu, c)


if __name__ == "__main__":
    # Sanity check 1: M/M/1 reproduce el ejemplo del peaje del apunte.
    m = mmc(lam=0.05, mu=1 / 12, c=1)
    print(f"[Peaje M/M/1] Lq={m.Lq:.2f} (esp ~0.91)  W={m.W:.1f}s (esp ~30.3)  "
          f"P0={m.P0:.2f} (esp ~0.40)")

    # Sanity check 2: M/M/c con c=1 == M/M/1.
    a = mmc(0.4, 0.5, 1)
    rho = 0.4 / 0.5
    print(f"[M/M/c c=1] Lq={a.Lq:.4f}  vs MM1 rho^2/(1-rho)={rho**2/(1-rho):.4f}")

    # Sanity check 3: M/M/c/K calcula bloqueo.
    k = mmck(0.4, 0.25, c=3, K=10)
    print(f"[M/M/c/K] Pb(fuga)={k.Pb_finite:.4%}  Lq={k.Lq:.3f}  rho={k.rho:.3f}")
