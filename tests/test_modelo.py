"""
test_modelo.py — Tests de validación del modelo (rigor matemático).

Ejecutar:  python -m pytest -q      (o)   python tests/test_modelo.py
No requiere pytest para correr standalone.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytical import mmc, mmck
from prng import exponencial_inversa, ExponentialStream, CongruentialGenerator
from config import Scenario
from simulation import correr


def aprox(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- Fórmulas analíticas ---------------------------------------------------- #
def test_mm1_peaje():
    """Reproduce el ejemplo del peaje del apunte (M/M/1).

    El apunte redondea μ≈0.083 (Lq≈0.91, W≈30.3); con el valor exacto μ=1/12 los
    valores son Lq=0.90 y W=30.0. Verificamos contra el valor exacto.
    """
    m = mmc(lam=0.05, mu=1 / 12, c=1)
    assert abs(m.Lq - 0.90) < 2e-2
    assert abs(m.W - 30.0) < 0.5
    assert abs(m.P0 - 0.40) < 1e-2


def test_mmc_reduce_a_mm1():
    """M/M/c con c=1 debe coincidir con la fórmula cerrada de M/M/1."""
    lam, mu = 0.4, 0.5
    m = mmc(lam, mu, 1)
    rho = lam / mu
    assert abs(m.Lq - rho ** 2 / (1 - rho)) < 1e-9
    assert abs(m.L - rho / (1 - rho)) < 1e-9


def test_mmc_estabilidad():
    inestable = mmc(1.0, 0.25, 2)      # rho = 2 > 1
    assert not inestable.stable
    assert math.isinf(inestable.Lq)


def test_mmck_bloqueo_en_rango():
    k = mmck(0.4, 0.25, c=3, K=10)
    assert 0.0 <= k.Pb_finite <= 1.0
    assert k.lambda_eff <= 0.4


def test_mmck_K_igual_c_es_erlang_b():
    """Con K=c (sin sala de espera) Pb es la fórmula de Erlang B."""
    lam, mu, c = 2.0, 1.0, 2
    a = lam / mu
    k = mmck(lam, mu, c, K=c)
    # Erlang B: B = (a^c/c!) / sum_{n=0}^{c} a^n/n!
    num = a ** c / math.factorial(c)
    den = sum(a ** n / math.factorial(n) for n in range(c + 1))
    assert abs(k.Pb_finite - num / den) < 1e-9


# --- PRNG / transformada inversa -------------------------------------------- #
def test_exponencial_inversa_formula():
    # x = -ln(R)/lam
    assert aprox(exponencial_inversa(math.e ** -1, 1.0), 1.0, 1e-12)


def test_exponencial_media_empirica():
    s = ExponentialStream(seed=1)
    lam = 0.3
    n = 100_000
    media = sum(s.sample(lam) for _ in range(n)) / n
    assert abs(media - 1 / lam) / (1 / lam) < 0.02      # < 2% error


def test_gcl_reproducible():
    g1 = CongruentialGenerator(seed=123)
    g2 = CongruentialGenerator(seed=123)
    assert [g1.next_float() for _ in range(50)] == [g2.next_float() for _ in range(50)]


# --- Simulación ------------------------------------------------------------- #
def test_sim_reproducible():
    sc = Scenario(sim_time=200.0, seed=99)
    r1 = correr(sc)
    r2 = correr(Scenario(sim_time=200.0, seed=99))
    assert (r1.spawned, r1.killed, r1.leaked) == (r2.spawned, r2.killed, r2.leaked)


def test_sim_balance_enemigos():
    sc = Scenario(sim_time=300.0)
    r = correr(sc)
    en_sistema = r.spawned - r.killed - r.leaked
    assert en_sistema >= 0       # no se pierden enemigos


def test_sim_little():
    """La simulación debe satisfacer aproximadamente la Ley de Little: L = λ_eff·W."""
    sc = Scenario(sim_time=4000.0, c=3, seed=7)
    r = correr(sc)
    L_sim = r.sys_area / sc.sim_time
    lam_eff = r.killed / sc.sim_time
    W_sim = r.sum_time_sys / r.killed
    assert abs(lam_eff * W_sim - L_sim) / L_sim < 0.1


def test_sim_concuerda_con_analitico_sin_temperatura():
    """Sin sobrecalentamiento (T_max enorme) y K grande, Wq sim ≈ Erlang C."""
    sc = Scenario(sim_time=20000.0, c=3, K=200, T_max=1e9, seed=5)
    r = correr(sc)
    wq_sim = r.sum_wait_q / r.killed
    wq_ana = mmc(sc.lam, sc.mu, sc.c).Wq
    assert abs(wq_sim - wq_ana) / wq_ana < 0.20    # < 20% (ruido estadístico)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            fallos += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fallos += 1; print(f"  ERROR {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - fallos}/{len(fns)} tests OK")
    sys.exit(1 if fallos else 0)
