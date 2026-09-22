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

from ..rng.lcg import MersenneTwisterGenerator, crear_generador_base
from ..rng.polar import PolarGenerator
from ..rng.rechazo import RechazoGenerator
from ..rng.truncada import InversaGenerator

SEMILLA = 13579
N_PLOT = 200_000
DIR_SALIDA = "resultados"


def _generadores():
    return {
        "Polar": PolarGenerator(MersenneTwisterGenerator(SEMILLA)),
        "Rechazo": RechazoGenerator(MersenneTwisterGenerator(SEMILLA)),
        "Inversa": InversaGenerator(MersenneTwisterGenerator(SEMILLA)),
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


def figura_reticulo(dir_salida=DIR_SALIDA, n_ternas=30000):
    """
    El reticulo de los generadores uniformes: la evidencia VISUAL de por que
    RANDU reprueba la prueba serial en 3D y los otros no.

    Cada punto es una terna consecutiva (u_i, u_{i+1}, u_{i+2}). Un generador
    sano llena el cubo de forma pareja. RANDU no: sus estados satisfacen
    exactamente

        x_{n+2} - 6*x_{n+1} + 9*x_n = 0   (mod m)

    asi que todas sus ternas viven en la familia de planos de normal
    (9, -6, 1), que resultan ser solo 15 dentro del cubo unitario.

    El angulo de camara no se elige a ojo: para ver los planos DE CANTO hay que
    mirar en una direccion contenida en ellos, es decir perpendicular a su
    normal. Con elevacion 0 la direccion de camara es (cos a, sin a, 0), y la
    condicion 9*cos(a) - 6*sin(a) = 0 da a = atan(9/6) = 56.31 grados. Desde
    cualquier otro angulo la nube de RANDU se ve tan sana como las otras dos,
    que es justamente por lo que este defecto sobrevivio una decada en
    produccion.
    """
    os.makedirs(dir_salida, exist_ok=True)
    nombres = ["mt", "minstd", "randu"]
    titulos = {"mt": "Mersenne Twister", "minstd": "LCG propio (MINSTD)",
               "randu": "RANDU (control defectuoso)"}

    azimut = float(np.degrees(np.arctan2(9.0, 6.0)))  # 56.31 grados

    fig = plt.figure(figsize=(13, 4.6))
    for i, nombre in enumerate(nombres, start=1):
        ternas = crear_generador_base(nombre, SEMILLA).uniform(3 * n_ternas).reshape(-1, 3)

        ax = fig.add_subplot(1, len(nombres), i, projection="3d")
        ax.scatter(ternas[:, 0], ternas[:, 1], ternas[:, 2],
                   s=0.35, alpha=0.35, color="#4c72b0", edgecolors="none")
        ax.view_init(elev=0, azim=azimut)
        ax.set_xlabel("$u_i$", fontsize=8)
        ax.set_ylabel("$u_{i+1}$", fontsize=8)
        ax.set_zlabel("$u_{i+2}$", fontsize=8)
        ax.set_title(titulos[nombre], fontsize=10)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Ternas consecutivas vistas desde azimut {azimut:.1f}$^\\circ$, "
                 "elevacion 0$^\\circ$: los planos de RANDU quedan de canto",
                 fontsize=12)
    fig.tight_layout()
    ruta = os.path.join(dir_salida, "reticulo_3d.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada: {ruta}")
    return ruta


if __name__ == "__main__":
    generar_figuras()
    figura_reticulo()
