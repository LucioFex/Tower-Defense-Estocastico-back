# Tower Defense Estocástico — Backend (SimPy + Análisis)

Repo 2 de 3 del TP de **Simulación de Sistemas** (UCEMA). Implementa la matemática del modelo de
colas y exporta `output.json` (el contrato que consume el frontend Godot). **Cero acoplamiento en
tiempo real**: Python calcula todo offline.

> Especificación del modelo, ecuaciones y contrato de datos: ver el repo
> `Tower-Defense-Estocastico-rag-karpathy`.

## Arquitectura del repo

```
config.py          Parámetros del escenario (variables de control)
prng.py            PRNG congruencial + transformada inversa exponencial (Unidad II)
analytical.py      Fórmulas cerradas M/M/1, M/M/c (Erlang C), M/M/c/K (bloqueo)
simulation.py      Módulo A — motor SimPy (M/M/c FIFO/K + temperatura continua)
exporter.py        Arma el output.json según el contrato + barrido (sweep) de c
validate_schema.py Auto-auditoría del contrato (8 invariantes) — sella la build
main.py            Módulo A — CLI: corre simulación + sweep y exporta output.json
analysis.py        Módulo B — lee output.json, genera gráficos e imprime conclusiones
experiments.py     Módulo B+ — réplicas+IC, escenario no estacionario, tipos de enemigo
tests/test_modelo.py  Tests de validación (fórmulas, PRNG, Little, sim vs analítico, extensiones)
```

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash;  Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Módulo A — Simulación y export
```bash
python main.py                       # escenario por defecto + sweep c=1..6 -> output.json
python main.py --c 4 --lam 0.5 --mu 0.25 --sim 1800 --seed 7
python main.py --no-sweep --out otra.json
```

### Módulo B — Análisis (gráficos + conclusiones)
```bash
python analysis.py                   # lee output.json -> figs/*.png + conclusiones en terminal
python analysis.py otra.json --outdir figs2
```

### Módulo B+ — Experimentos avanzados
```bash
python experiments.py                # los 3 estudios -> figs_exp/*.png + conclusiones
python experiments.py --reps 20 --only ci
```
1. **Réplicas + IC 95%** (reducción de varianza; `c*` robusto sobre N semillas).
2. **No estacionario** (oleadas λ(t)): demuestra *dimensionar al pico, no al promedio*.
3. **Tipos de enemigo** (V.A. discreta): a igual media de servicio, más varianza ⇒ más fuga
   (Pollaczek-Khinchine). Activable también vía `Scenario(enemy_types=..., lam_schedule=...)`.

### Tests / auditoría
```bash
python tests/test_modelo.py          # 15 tests (sin pytest) o: python -m pytest -q
python validate_schema.py output.json
```

## El modelo en una frase

Cada **torre = servidor** M/M/c (cola FIFO única vía `simpy.Store`); **enemigo = cliente**;
**fuga = bloqueo** por capacidad finita `K`. La **temperatura** (variable continua, EDO por tramos:
rampa al disparar, enfriamiento de Newton al descansar) apaga torres sobrecalentadas, reduciendo la
capacidad efectiva y **rompiendo** el supuesto markoviano — por eso la simulación aporta sobre la
fórmula. Ver RAG.

## Resultados clave (escenario por defecto: λ=0.4, μ=0.25, c=3, K=10)

- `c_min = 2` (mínimo para estabilidad ρ<1); `c* = 4` (óptimo económico torres+fugas).
- La 1ª torre sobre `c_min` captura **~89%** de la reducción total de cola →
  **rendimiento marginal decreciente** (ΔLq por torre ≈ [2.5, 0.25, 0.05, 0.01]).
- **Ley de Little** verificada (error ~1.6%) → modelo consistente.
- Fuga simulada (~6.6%) ≫ analítica M/M/c/K (~0.16%): la brecha la explica la
  **indisponibilidad por temperatura** (validación de la hipótesis H3).

## Salida

- `output.json` — contrato v1.0 (consumido por Godot y por `analysis.py`).
- `figs/` — `01_series_colas`, `02_temperatura`, `03_utilizacion`,
  `04_rendimiento_marginal`, `05_costo_optimo`.

## Pensado para Ubuntu de bajos recursos

Sin solver numérico (la temperatura se integra en forma cerrada por tramos: O(eventos)). Solo
`simpy`, `matplotlib`, `pandas`. El export es un único JSON estático.
