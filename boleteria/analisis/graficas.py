"""
Graficas de validacion para el informe: Q-Q plots e histogramas con densidad
teorica superpuesta, para el metodo Polar y el de Rechazo.

Un Q-Q plot enfrenta los cuantiles teoricos de la distribucion objetivo contra
los cuantiles empiricos de la muestra; si el generador es correcto, los puntos
caen sobre la recta identidad (la diagonal roja). Las colas son justamente donde
se ven las desviaciones, asi que es el mejor complemento visual de KS y Chi2.

Uso:
    python -m boleteria.analisis.graficas
"""
import os
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..rng.lcg import MersenneTwisterGenerator
from ..rng.polar import PolarGenerator
from ..rng.rechazo import RechazoGenerator

SEMILLA = 13579
N_PLOT = 200_000
DIR_SALIDA = "resultados"


def _generadores():
    return {
        "Polar": PolarGenerator(MersenneTwisterGenerator(SEMILLA)),
        "Rechazo": RechazoGenerator(MersenneTwisterGenerator(SEMILLA)),
    }


def _casos(gen):
    """(etiqueta, muestra, distribucion teorica scipy) para cada generador."""
    return [
        ("N(0,1) estandar",
         gen.normal(N_PLOT, 0.0, 1.0),
         stats.norm(0.0, 1.0)),
        ("Truncada valuacion\nN(500,220) en [50, inf)",
         gen.normal_truncada(N_PLOT, 500.0, 220.0, 50.0, np.inf),
         stats.truncnorm((50.0 - 500.0) / 220.0, np.inf, loc=500.0, scale=220.0)),
        ("Truncada checkout\nN(90,30) en [15, 600]",
         gen.normal_truncada(N_PLOT, 90.0, 30.0, 15.0, 600.0),
         stats.truncnorm((15.0 - 90.0) / 30.0, (600.0 - 90.0) / 30.0, loc=90.0, scale=30.0)),
    ]


def qq_plot(ax, muestra, dist, n_puntos=3000):
    """Q-Q plot con posiciones de ploteo (i-0.5)/n submuestreadas para no saturar."""
    n = muestra.size
    idx = np.unique(np.linspace(0, n - 1, min(n, n_puntos)).astype(int))
    empiricos = np.sort(muestra)[idx]
    teoricos = dist.ppf((idx + 0.5) / n)

    ax.scatter(teoricos, empiricos, s=6, alpha=0.45, color="#4c72b0", edgecolors="none")
    lo = float(min(teoricos[0], empiricos[0]))
    hi = float(max(teoricos[-1], empiricos[-1]))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="identidad")
    ax.set_xlabel("Cuantiles teoricos")
    ax.set_ylabel("Cuantiles empiricos")
    ax.legend(loc="upper left", fontsize=8)


def histograma(ax, muestra, dist, bins=80):
    """Histograma normalizado con la densidad teorica superpuesta."""
    ax.hist(muestra, bins=bins, density=True, color="#55a868", alpha=0.6, edgecolor="none")
    lo, hi = muestra.min(), muestra.max()
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, dist.pdf(xs), "r-", lw=1.5, label="densidad teorica")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper right", fontsize=8)


def generar_figuras(dir_salida=DIR_SALIDA):
    """Una figura por generador: filas = casos, columnas = [histograma, Q-Q]."""
    os.makedirs(dir_salida, exist_ok=True)
    rutas = []
    for nombre, gen in _generadores().items():
        casos = _casos(gen)
        fig, axes = plt.subplots(len(casos), 2, figsize=(11, 3.4 * len(casos)))
        for fila, (etiqueta, muestra, dist) in enumerate(casos):
            histograma(axes[fila, 0], muestra, dist)
            qq_plot(axes[fila, 1], muestra, dist)
            axes[fila, 0].set_title(f"{etiqueta}  -  histograma", fontsize=9)
            axes[fila, 1].set_title(f"{etiqueta}  -  Q-Q plot", fontsize=9)

        fig.suptitle(f"Validacion visual - Generador {nombre}", fontsize=13, y=1.0)
        fig.tight_layout()
        ruta = os.path.join(dir_salida, f"qq_hist_{nombre.lower()}.png")
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        plt.close(fig)
        rutas.append(ruta)
        print(f"Figura guardada: {ruta}")
    return rutas


if __name__ == "__main__":
    generar_figuras()
