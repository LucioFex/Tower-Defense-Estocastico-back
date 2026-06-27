"""
prng.py — Generadores de números pseudoaleatorios y transformada inversa.

Unidad II (Simulación de Sistemas). Cumple explícitamente:
  - Generador Congruencial Lineal (GCL) y su variante multiplicativa.
  - Transformada inversa para la distribución Exponencial negativa.

Decisión de diseño (ver RAG/04_generadores_aleatorios.md): la SIMULACIÓN usa el
Mersenne Twister de `random` (período 2^19937-1) sembrado para reproducibilidad,
y aplica la transformada inversa documentada aquí para obtener las exponenciales.
El GCL queda implementado y testeado como cumplimiento académico de la Unidad II.
"""
from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------- #
#  Generador Congruencial Lineal (Unidad II)                                   #
# --------------------------------------------------------------------------- #
class CongruentialGenerator:
    """X_{i+1} = (a * X_i + C) mod m   ;   R_i = X_i / m  ->  U[0,1).

    Con C = 0 se obtiene la variante MULTIPLICATIVA (más rápida).
    Parámetros por defecto: los de glibc (período completo m = 2^31).
    """

    def __init__(self, seed: int = 1, a: int = 1103515245, c: int = 12345,
                 m: int = 2 ** 31):
        if not (m > a and m > c and m > 0):
            raise ValueError("Debe cumplirse m > a, m > C y m > 0")
        self.a, self.c, self.m = a, c, m
        self.state = seed % m

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self) -> float:
        """Número aleatorio R en [0, 1)."""
        return self.next_int() / self.m


def multiplicative_generator(seed: int, a: int, m: int) -> CongruentialGenerator:
    """Variante multiplicativa de congruencias (C = 0)."""
    return CongruentialGenerator(seed=seed, a=a, c=0, m=m)


# --------------------------------------------------------------------------- #
#  Transformada inversa: Exponencial negativa (Unidad II)                       #
# --------------------------------------------------------------------------- #
def exponencial_inversa(r: float, lam: float) -> float:
    """Genera una muestra Exp(lam) por transformada inversa.

        F(x) = 1 - e^{-lam*x}  ->  x = -(1/lam) * ln(1 - R)
    Se usa la forma equivalente x = -(1/lam) * ln(R) (R y 1-R son ambos U[0,1)).

    `r` debe estar en (0, 1].  `lam` es la tasa (1/media).
    """
    if not (0.0 < r <= 1.0):
        raise ValueError("r debe estar en (0, 1]")
    if lam <= 0.0:
        raise ValueError("lam debe ser > 0")
    return -math.log(r) / lam


class ExponentialStream:
    """Flujo reproducible de muestras Exp(lam) usando Mersenne + transformada inversa.

    Encapsula la decisión documentada: uniforme de calidad (Mersenne) + el método
    de generación de la materia (transformada inversa).
    """

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def sample(self, lam: float) -> float:
        # random() ∈ [0,1); evitamos el 0 para no romper el ln.
        r = 1.0 - self._rng.random()  # r ∈ (0, 1]
        return exponencial_inversa(r, lam)

    def uniform(self) -> float:
        return self._rng.random()


if __name__ == "__main__":
    # Validación rápida: media empírica vs. teórica (1/lam).
    lam = 0.25
    stream = ExponentialStream(seed=42)
    n = 200_000
    muestras = [stream.sample(lam) for _ in range(n)]
    media = sum(muestras) / n
    print(f"Exp(lam={lam}) -> media empírica={media:.4f}  teórica={1/lam:.4f}")

    gcl = CongruentialGenerator(seed=7)
    us = [gcl.next_float() for _ in range(100_000)]
    print(f"GCL -> media uniforme={sum(us)/len(us):.4f} (esperada ~0.5)")
