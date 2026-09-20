"""
Validacion estadistica de los generadores normales: pruebas de bondad de ajuste.

Se aplican dos pruebas complementarias:

  * Kolmogorov-Smirnov (KS): sobre SUBMUESTRAS de n = 20,000. Con los millones de
    datos que produce el benchmark, KS rechaza casi cualquier generador porque su
    potencia crece con n y detecta desviaciones infimas e irrelevantes (falso
    rechazo). Submuestrear a 20,000 y repetir la prueba varias veces da una lectura
    honesta: reportamos el estadistico y el valor-p medianos y la fraccion de
    submuestras que pasan a un nivel alpha.

  * Chi-cuadrado: con celdas equiprobables bajo la hipotesis nula (frecuencia
    esperada igual en cada celda), que es la particion recomendada.

Uso:
    python -m boleteria.analisis.validacion
"""
import numpy as np
from scipy import stats

from ..rng.lcg import MersenneTwisterGenerator
from ..rng.polar import PolarGenerator
from ..rng.rechazo import RechazoGenerator

SEMILLA = 7654321
N_SUBMUESTRA = 20_000   # tamano fijado por la rubrica para evitar el falso rechazo
N_POOL = 1_000_000      # muestra grande de la que se extraen las submuestras
ALPHA = 0.05


def _generadores():
    return {
        "Polar": PolarGenerator(MersenneTwisterGenerator(SEMILLA)),
        "Rechazo": RechazoGenerator(MersenneTwisterGenerator(SEMILLA)),
    }


def prueba_ks(muestra, dist, n_sub=N_SUBMUESTRA, n_rep=30, alpha=ALPHA, rng=None):
    """
    KS repetido sobre submuestras de tamano n_sub extraidas de `muestra`.

    Devuelve el estadistico y valor-p medianos y la fraccion de submuestras que
    NO se rechazan (p > alpha). `dist` es una distribucion congelada de scipy
    (p. ej. stats.norm(mu, sigma)) con los parametros teoricos conocidos.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    estadisticos, valores_p = [], []
    for _ in range(n_rep):
        sub = rng.choice(muestra, size=n_sub, replace=False)
        res = stats.kstest(sub, dist.cdf)
        estadisticos.append(res.statistic)
        valores_p.append(res.pvalue)

    valores_p = np.asarray(valores_p)
    return {
        "estadistico_mediano": float(np.median(estadisticos)),
        "pvalor_mediano": float(np.median(valores_p)),
        "fraccion_pasa": float(np.mean(valores_p > alpha)),
        "n_sub": n_sub,
        "n_rep": n_rep,
    }


def prueba_chi2(muestra, dist, k_bins=40, alpha=ALPHA):
    """
    Chi-cuadrado con celdas equiprobables bajo la nula.

    Los bordes se toman como cuantiles de la distribucion teorica, de modo que
    la frecuencia esperada es identica (n/k) en cada celda. Como los parametros
    son conocidos, los grados de libertad son k - 1.
    """
    muestra = np.asarray(muestra)
    n = muestra.size
    bordes = dist.ppf(np.linspace(0.0, 1.0, k_bins + 1))
    bordes[0], bordes[-1] = -np.inf, np.inf

    observadas, _ = np.histogram(muestra, bins=bordes)
    esperadas = np.full(k_bins, n / k_bins)

    estadistico = float(np.sum((observadas - esperadas) ** 2 / esperadas))
    gl = k_bins - 1
    pvalor = float(stats.chi2.sf(estadistico, gl))
    return {
        "estadistico": estadistico,
        "pvalor": pvalor,
        "gl": gl,
        "k_bins": k_bins,
        "pasa": pvalor > alpha,
    }


def _casos_prueba(gen):
    """Distribuciones teoricas a validar para cada generador."""
    return [
        # (etiqueta, muestra, distribucion teorica scipy)
        ("N(0,1) estandar",
         gen.normal(N_POOL, 0.0, 1.0),
         stats.norm(0.0, 1.0)),
        ("Truncada valuacion N(500,220) en [50, inf)",
         gen.normal_truncada(N_POOL, 500.0, 220.0, 50.0, np.inf),
         stats.truncnorm((50.0 - 500.0) / 220.0, np.inf, loc=500.0, scale=220.0)),
        ("Truncada checkout N(90,30) en [15, 600]",
         gen.normal_truncada(N_POOL, 90.0, 30.0, 15.0, 600.0),
         stats.truncnorm((15.0 - 90.0) / 30.0, (600.0 - 90.0) / 30.0, loc=90.0, scale=30.0)),
    ]


def reporte_bondad():
    rng = np.random.default_rng(2024)
    print("\n=== Pruebas de bondad de ajuste ===")
    print(f"Pool = {N_POOL:,} | KS sobre submuestras de {N_SUBMUESTRA:,} | alpha = {ALPHA}\n")

    for nombre, gen in _generadores().items():
        print(f"--- Generador {nombre} ---")
        for etiqueta, muestra, dist in _casos_prueba(gen):
            ks = prueba_ks(muestra, dist, rng=rng)
            chi = prueba_chi2(muestra, dist)
            print(f"  {etiqueta}")
            print(f"    KS  : D_mediano={ks['estadistico_mediano']:.4f} "
                  f"p_mediano={ks['pvalor_mediano']:.3f} "
                  f"pasan={ks['fraccion_pasa']*100:.0f}% de {ks['n_rep']} submuestras")
            print(f"    Chi2: X2={chi['estadistico']:.1f} gl={chi['gl']} "
                  f"p={chi['pvalor']:.3f} -> {'PASA' if chi['pasa'] else 'RECHAZA'}")
        print()


if __name__ == "__main__":
    reporte_bondad()
