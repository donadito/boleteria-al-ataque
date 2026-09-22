"""
Validacion estadistica en TRES CAPAS.

La idea es que cada capa solo tiene sentido si la de abajo pasa: unas normales
impecables construidas sobre uniformes malos siguen siendo malas, y un modelo
bien programado alimentado con normales mal generadas produce numeros bonitos y
falsos. Por eso se prueban en orden.

  CAPA 1 - El generador uniforme base.
      Chi-cuadrado de uniformidad, prueba de rachas (independencia de signo),
      autocorrelacion a varios rezagos con Ljung-Box, y la prueba SERIAL EN 3D.
      La serial en 3D es la importante: hay generadores que pasan las tres
      primeras y aun asi son inservibles, porque sus defectos solo se ven en
      dimensiones mayores. RANDU se incluye como control defectuoso justamente
      para demostrar que este banco de pruebas tiene poder para detectarlo.

  CAPA 2 - Las distribuciones generadas sobre esos uniformes.
      Kolmogorov-Smirnov sobre SUBMUESTRAS de n = 20,000. Con los millones de
      datos que produce el benchmark, KS rechaza casi cualquier generador
      porque su potencia crece con n y detecta desviaciones infimas e
      irrelevantes (falso rechazo). Submuestrear y repetir da una lectura
      honesta: se reportan el estadistico y el valor-p medianos y la fraccion
      de submuestras que pasan. Se acompana de chi-cuadrado con celdas
      equiprobables bajo la nula, que es la particion recomendada.

  CAPA 3 - El modelo completo contra solucion analitica cerrada.
      El proceso de llegadas es un Poisson no homogeneo generado por
      adelgazamiento, y eso tiene dos propiedades exactas que se pueden
      verificar sin simular nada:
        (a) el numero de llegadas en [a,b] es Poisson con media la integral de
            lambda sobre ese intervalo;
        (b) condicionado a N llegadas, los tiempos se distribuyen como iid con
            densidad proporcional a lambda(t).
      Si el adelgazamiento estuviera mal implementado (mayorante mal calculada,
      comparacion invertida), (a) fallaria en la media y (b) en la forma.

Uso:
    python -m boleteria.analisis.validacion
"""
import numpy as np
from scipy import integrate, stats

from ..rng.lcg import PRESETS, crear_generador_base
from ..rng.poisson import LlegadasNHPP
from ..rng.polar import PolarGenerator
from ..rng.rechazo import RechazoGenerator
from ..rng.truncada import InversaGenerator

SEMILLA = 7654321
N_SUBMUESTRA = 20_000   # tamano fijado para evitar el falso rechazo de KS
N_POOL = 1_000_000      # muestra grande de la que se extraen las submuestras
N_UNIFORMES = 500_000   # muestra para las pruebas de la capa 1
ALPHA = 0.05


def _generadores():
    return {
        "Polar": PolarGenerator(crear_generador_base("mt", SEMILLA)),
        "Rechazo": RechazoGenerator(crear_generador_base("mt", SEMILLA)),
        "Inversa": InversaGenerator(crear_generador_base("mt", SEMILLA)),
    }


# ===========================================================================
# CAPA 1: el generador uniforme base
# ===========================================================================

def prueba_uniformidad(u, k_bins=100, alpha=ALPHA):
    """Chi-cuadrado sobre k celdas de igual ancho en [0,1)."""
    observadas, _ = np.histogram(u, bins=k_bins, range=(0.0, 1.0))
    esperadas = np.full(k_bins, u.size / k_bins)
    estadistico = float(np.sum((observadas - esperadas) ** 2 / esperadas))
    gl = k_bins - 1
    pvalor = float(stats.chi2.sf(estadistico, gl))
    return {"prueba": "Chi2 uniformidad", "estadistico": estadistico,
            "gl": gl, "pvalor": pvalor, "pasa": pvalor > alpha}


def prueba_rachas(u, alpha=ALPHA):
    """
    Rachas por encima/debajo de la mediana teorica (0.5).

    Cuenta cuantos bloques consecutivos del mismo signo hay. Muy pocas rachas
    delata tendencia o agrupamiento; demasiadas, alternancia sistematica. Bajo
    independencia, R es asintoticamente normal con la media y varianza de
    Wald-Wolfowitz.
    """
    signos = u > 0.5
    n1 = int(np.count_nonzero(signos))
    n2 = int(signos.size - n1)
    if n1 == 0 or n2 == 0:
        return {"prueba": "Rachas", "estadistico": float("inf"),
                "gl": None, "pvalor": 0.0, "pasa": False}

    rachas = int(1 + np.count_nonzero(signos[1:] != signos[:-1]))
    n = n1 + n2
    media = 2.0 * n1 * n2 / n + 1.0
    varianza = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    z = (rachas - media) / np.sqrt(varianza)
    pvalor = float(2 * stats.norm.sf(abs(z)))
    return {"prueba": "Rachas", "estadistico": float(z), "gl": None,
            "pvalor": pvalor, "pasa": pvalor > alpha, "rachas": rachas,
            "rachas_esperadas": float(media)}


def prueba_autocorrelacion(u, max_rezago=10, alpha=ALPHA):
    """
    Autocorrelacion a los rezagos 1..max_rezago, resumida con Ljung-Box.

    Bajo independencia cada rho_k es aproximadamente N(0, 1/n), y el
    estadistico Q de Ljung-Box agrega todos los rezagos en una sola prueba
    chi-cuadrado con max_rezago grados de libertad.
    """
    x = u - u.mean()
    n = x.size
    denominador = float(np.dot(x, x))

    rhos = []
    for k in range(1, max_rezago + 1):
        rhos.append(float(np.dot(x[:-k], x[k:]) / denominador))
    rhos = np.array(rhos)

    q = n * (n + 2) * np.sum(rhos ** 2 / (n - np.arange(1, max_rezago + 1)))
    pvalor = float(stats.chi2.sf(q, max_rezago))
    return {"prueba": f"Ljung-Box (rezagos 1-{max_rezago})", "estadistico": float(q),
            "gl": max_rezago, "pvalor": pvalor, "pasa": pvalor > alpha,
            "rho_max": float(np.max(np.abs(rhos))),
            "rho_lag1": float(rhos[0])}


def prueba_serial_3d(u, k=10, alpha=ALPHA):
    """
    Prueba serial en tres dimensiones: chi-cuadrado sobre el cubo unitario.

    Se arman ternas NO solapadas (u0,u1,u2), (u3,u4,u5), ... y se cuenta cuantas
    caen en cada una de las k^3 celdas del cubo. Bajo independencia el conteo
    deberia repartirse parejo.

    Esta es la prueba que separa un generador usable de uno inservible. Un LCG
    con malos parametros produce ternas que viven en unos pocos hiperplanos del
    cubo: en una dimension se ve perfectamente uniforme, pero en tres deja
    huecos enormes. RANDU es el caso de libro (sus ternas caen en 15 planos) y
    aqui debe reprobar de forma escandalosa.
    """
    m = (u.size // 3) * 3
    ternas = u[:m].reshape(-1, 3)
    indices = np.minimum((ternas * k).astype(np.int64), k - 1)
    planos = indices[:, 0] * k * k + indices[:, 1] * k + indices[:, 2]

    observadas = np.bincount(planos, minlength=k ** 3)
    esperadas = ternas.shape[0] / (k ** 3)
    estadistico = float(np.sum((observadas - esperadas) ** 2 / esperadas))
    gl = k ** 3 - 1
    pvalor = float(stats.chi2.sf(estadistico, gl))
    return {"prueba": f"Serial 3D ({k}^3 celdas)", "estadistico": estadistico,
            "gl": gl, "pvalor": pvalor, "pasa": pvalor > alpha,
            "celdas_vacias": int(np.count_nonzero(observadas == 0))}


def reporte_capa1(n=N_UNIFORMES, alpha=ALPHA):
    """Corre el banco completo de la capa 1 sobre cada generador base."""
    print("\n" + "=" * 78)
    print("CAPA 1 - Generadores uniformes base")
    print("=" * 78)
    print(f"n = {n:,} uniformes por generador | alpha = {alpha}\n")

    resultados = {}
    for nombre in ["mt"] + list(PRESETS):
        gen = crear_generador_base(nombre, SEMILLA)
        u = gen.uniform(n)
        pruebas = [
            prueba_uniformidad(u),
            prueba_rachas(u),
            prueba_autocorrelacion(u),
            prueba_serial_3d(u),
        ]
        resultados[nombre] = pruebas

        etiqueta = "Mersenne Twister" if nombre == "mt" else PRESETS[nombre][3]
        print(f"--- {nombre.upper()}: {etiqueta}")
        for p in pruebas:
            veredicto = "PASA" if p["pasa"] else "RECHAZA"
            gl = f"gl={p['gl']}" if p["gl"] else ""
            print(f"    {p['prueba']:26s} estadistico={p['estadistico']:12.2f} {gl:8s} "
                  f"p={p['pvalor']:.3e}  -> {veredicto}")
        fallos = [p["prueba"] for p in pruebas if not p["pasa"]]
        if fallos:
            print(f"    !! Reprueba: {', '.join(fallos)}")
        print()

    return resultados


# ===========================================================================
# CAPA 2: las distribuciones generadas
# ===========================================================================

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
        # (etiqueta, muestra, distribucion teorica)
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
    print("\n" + "=" * 78)
    print("CAPA 2 - Distribuciones generadas")
    print("=" * 78)
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


# ===========================================================================
# CAPA 3: el modelo completo contra solucion analitica
# ===========================================================================

def reporte_nhpp(n_corridas=300, alpha=ALPHA):
    """
    Valida el proceso de llegadas contra sus dos propiedades exactas.

    (a) Conteo. E[N(a,b)] = integral de lambda entre a y b, y N es Poisson, asi
        que Var[N] = E[N]. Se comparan media y varianza muestrales contra la
        integral numerica, y se prueba el ajuste Poisson con chi-cuadrado.

    (b) Forma. Condicionado a N, los tiempos son iid con densidad
        lambda(t)/integral. Se prueba con KS contra esa CDF, construida por
        integracion numerica acumulada.
    """
    print("\n" + "=" * 78)
    print("CAPA 3 - Modelo de llegadas contra solucion analitica")
    print("=" * 78)

    nhpp = LlegadasNHPP(crear_generador_base("mt", SEMILLA))
    apertura, duracion, pool = 600.0, 240.0, 60_000
    intensidad = lambda t: nhpp.intensidad(t, apertura, pool)

    esperado, _ = integrate.quad(intensidad, apertura, apertura + duracion, limit=400)
    print(f"\nVentana [{apertura:.0f}, {apertura + duracion:.0f}] min, pool = {pool:,}")
    print(f"  Integral de lambda (esperado analitico) : {esperado:,.1f} llegadas")

    conteos, todas = [], []
    for _ in range(n_corridas):
        tiempos = nhpp.generar_fase(apertura, duracion, pool)
        conteos.append(len(tiempos))
        todas.extend(tiempos)
    conteos = np.array(conteos, dtype=float)

    media, varianza = conteos.mean(), conteos.var(ddof=1)
    error_est = conteos.std(ddof=1) / np.sqrt(n_corridas)
    z = (media - esperado) / error_est
    p_media = float(2 * stats.norm.sf(abs(z)))

    print(f"  Media simulada ({n_corridas} corridas)        : {media:,.1f} "
          f"(EE {error_est:,.1f})")
    print(f"  Error relativo                          : {abs(media - esperado) / esperado:.3%}")
    print(f"  z de la media contra la integral        : {z:+.2f}  p={p_media:.3f} "
          f"-> {'PASA' if p_media > alpha else 'RECHAZA'}")
    # Prueba de dispersion: para una Poisson, (n-1)*s^2/media ~ chi2(n-1).
    # Sin esta prueba el indice se lee mal: con 300 corridas su desviacion
    # estandar es sqrt(2/(n-1)) ~ 0.08, asi que un 0.91 esta dentro del ruido.
    indice = varianza / media
    d_disp = (n_corridas - 1) * varianza / media
    p_disp = float(2 * min(stats.chi2.cdf(d_disp, n_corridas - 1),
                           stats.chi2.sf(d_disp, n_corridas - 1)))
    print(f"  Indice de dispersion Var/Media          : {indice:.3f} "
          f"+/- {np.sqrt(2 / (n_corridas - 1)):.3f} (Poisson => 1.0)")
    print(f"  Prueba de dispersion chi2               : D={d_disp:.1f} "
          f"gl={n_corridas - 1} p={p_disp:.3f} -> {'PASA' if p_disp > alpha else 'RECHAZA'}")

    # (b) Forma condicional: KS contra la CDF normalizada de lambda.
    rejilla = np.linspace(apertura, apertura + duracion, 2001)
    valores = np.array([intensidad(t) for t in rejilla])
    acumulada = integrate.cumulative_trapezoid(valores, rejilla, initial=0.0)
    acumulada /= acumulada[-1]
    cdf = lambda t: np.interp(t, rejilla, acumulada)

    muestra = np.array(todas)
    if muestra.size > 50_000:  # KS con millones de puntos rechaza por nada
        muestra = np.random.default_rng(1).choice(muestra, 50_000, replace=False)
    res = stats.kstest(muestra, cdf)
    print(f"  KS de los tiempos contra lambda(t)/Int  : D={res.statistic:.4f} "
          f"p={res.pvalue:.3f} -> {'PASA' if res.pvalue > alpha else 'RECHAZA'}")
    print()


def main():
    reporte_capa1()
    reporte_bondad()
    reporte_nhpp()


if __name__ == "__main__":
    main()
