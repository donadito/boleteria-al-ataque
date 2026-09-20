"""
Benchmark de velocidad de los generadores normales a escala grande (4x10^6).

Compara, para el metodo Polar y el de Rechazo, el regimen ESCALAR (bucle
muestra a muestra) contra el VECTORIZADO (rechazo por lotes sobre arreglos de
Numpy). Reporta tiempos, speedup, uniformes consumidos por normal y ordena los
metodos en cada regimen para verificar de forma explicita si el ranking de
velocidad se invierte al pasar de escalar a vectorizado.

Uso:
    python -m boleteria.analisis.benchmark
"""
import time
import os
import numpy as np

from ..rng.lcg import MersenneTwisterGenerator
from ..rng.polar import PolarGenerator
from ..rng.rechazo import RechazoGenerator

N_DEFECTO = 4_000_000
SEMILLA = 20240501

METODOS = {
    "Polar": PolarGenerator,
    "Rechazo": RechazoGenerator,
}


def _cronometrar(func, repeticiones: int) -> float:
    """Devuelve el mejor tiempo (segundos) de varias corridas, para amortiguar el ruido."""
    mejor = float("inf")
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        func()
        mejor = min(mejor, time.perf_counter() - inicio)
    return mejor


def _uniformes_por_normal(Gen, n: int = 200_000) -> float:
    """Estima cuantos uniformes consume el metodo por cada normal producida."""
    class _Contador(MersenneTwisterGenerator):
        def __init__(self, seed):
            super().__init__(seed)
            self.consumidos = 0

        def uniform(self, m: int) -> np.ndarray:
            self.consumidos += m
            return super().uniform(m)

    contador = _Contador(SEMILLA)
    Gen(contador).normal(n, 0.0, 1.0)
    return contador.consumidos / n


def correr_benchmark(n: int = N_DEFECTO, repeticiones: int = 3) -> dict:
    """Ejecuta el benchmark completo y devuelve un diccionario con los resultados."""
    resultados = {}
    for nombre, Gen in METODOS.items():
        # Regimen escalar (bucle muestra a muestra).
        gen_esc = Gen(MersenneTwisterGenerator(SEMILLA))
        t_escalar = _cronometrar(lambda g=gen_esc: g.normal_escalar(n), repeticiones)

        # Regimen vectorizado (rechazo por lotes).
        gen_vec = Gen(MersenneTwisterGenerator(SEMILLA))
        t_vector = _cronometrar(lambda g=gen_vec: g.normal(n), repeticiones)

        resultados[nombre] = {
            "escalar": t_escalar,
            "vectorizado": t_vector,
            "speedup": t_escalar / t_vector,
            "uniformes_por_normal": _uniformes_por_normal(Gen),
        }
    return resultados


def _ranking(resultados: dict, regimen: str) -> list:
    """Ordena los metodos del mas rapido al mas lento en un regimen dado."""
    return sorted(resultados, key=lambda m: resultados[m][regimen])


def imprimir_reporte(resultados: dict, n: int) -> None:
    print(f"\n=== Benchmark de generadores normales (n = {n:,}) ===\n")
    encabezado = f"{'Metodo':10s} {'Escalar (s)':>13s} {'Vectorizado (s)':>17s} {'Speedup':>9s} {'Unif/normal':>13s}"
    print(encabezado)
    print("-" * len(encabezado))
    for nombre, r in resultados.items():
        print(f"{nombre:10s} {r['escalar']:13.3f} {r['vectorizado']:17.4f} "
              f"{r['speedup']:8.1f}x {r['uniformes_por_normal']:13.2f}")

    rank_esc = _ranking(resultados, "escalar")
    rank_vec = _ranking(resultados, "vectorizado")
    print("\nRanking por velocidad (mas rapido -> mas lento):")
    print(f"  Escalar:     {' > '.join(rank_esc)}")
    print(f"  Vectorizado: {' > '.join(rank_vec)}")
    if rank_esc != rank_vec:
        print("  => El ranking SE INVIERTE al vectorizar.")
    else:
        print("  => El ranking se mantiene: el mismo metodo domina en ambos regimenes.")
        print(f"     ({rank_vec[0]} gana porque consume menos uniformes por normal).")


def graficar(resultados: dict, n: int, ruta: str = "resultados/benchmark_4M.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    metodos = list(resultados)
    escalar = [resultados[m]["escalar"] for m in metodos]
    vector = [resultados[m]["vectorizado"] for m in metodos]

    x = np.arange(len(metodos))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - ancho / 2, escalar, ancho, label="Escalar", color="#c44e52")
    b2 = ax.bar(x + ancho / 2, vector, ancho, label="Vectorizado", color="#4c72b0")

    ax.set_ylabel("Tiempo (s)")
    ax.set_title(f"Generacion de {n:,} normales: escalar vs vectorizado")
    ax.set_xticks(x)
    ax.set_xticklabels(metodos)
    ax.legend()
    for barras in (b1, b2):
        ax.bar_label(barras, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def main():
    n = N_DEFECTO
    resultados = correr_benchmark(n)
    imprimir_reporte(resultados, n)
    ruta = graficar(resultados, n)
    print(f"\nGrafica guardada en: {ruta}")


if __name__ == "__main__":
    main()
