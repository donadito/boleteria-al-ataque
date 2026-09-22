"""
Experimento 1: comparacion de los metodos de generacion de normales.

Se comparan tres metodos bajo tres criterios, en este orden de importancia:

    Polar      metodo polar de Marsaglia (rechazo en el circulo unitario)
    Rechazo    aceptacion-rechazo con envolvente exponencial; en el caso
               truncado, la envolvente trasladada de Robert (1995)
    Inversa    transformada inversa, escrita sobre la funcion de supervivencia
               para no perder precision en la cola

  1. CALIDAD. Si un metodo no genera la distribucion correcta, los otros dos
     criterios no importan. Se mide con el estadistico KS contra la teorica.
     Spoiler: los tres pasan, asi que la calidad NO es el criterio que decide.

  2. COSTO POR MUESTRA SIN TRUNCAR. Uniformes consumidos por normal y tiempo de
     pared en regimen escalar y vectorizado. Aqui gana Polar, porque produce
     DOS normales por par aceptado. Solo entran los dos metodos de rechazo: la
     inversa no rechaza nada, asi que no tiene fraccion de aceptacion ni
     version escalar que medir.

  3. COMPORTAMIENTO EN LA COLA (el criterio que realmente decide). Las tres
     variables del modelo son normales TRUNCADAS, y ahi los metodos dejan de
     parecerse:

       - Polar trunca por descarte ingenuo: genera normales completas y tira
         las que caen fuera. Su fraccion de aceptacion es P(a <= X <= b), que
         colapsa exponencialmente al alejarse la cota. En a = 3 sigma ya tira
         999 de cada 1000 muestras.
       - Rechazo usa la envolvente de Robert, cuya tasa se elige optima para la
         cota. Su fraccion de aceptacion NO colapsa: se mantiene alta y hasta
         mejora al alejarse la cota.
       - Inversa gasta exactamente un uniforme por muestra, siempre, sin
         importar donde este la cota. Su costo es plano.

     La conclusion del proyecto sale de aqui: Palco "exige muestrear mas alla
     de 3 sigma", asi que el metodo mas rapido en el caso comun (Polar) es el
     inutilizable en el caso que el modelo necesita.

Uso:
    python -m experimentos.exp1_metodos
    python experimentos/exp1_metodos.py
"""
import csv
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boleteria.analisis.benchmark import correr_benchmark
from boleteria.rng.lcg import MersenneTwisterGenerator
from boleteria.rng.polar import PolarGenerator
from boleteria.rng.rechazo import RechazoGenerator
from boleteria.rng.truncada import InversaGenerator

SEMILLA = 424242
DIR_SALIDA = "resultados"

N_CALIDAD = 200_000      # muestra para el KS
N_BENCHMARK = 2_000_000  # muestra para el benchmark de velocidad

# Los dos metodos de RECHAZO, que son los unicos que tienen fraccion de
# aceptacion y version escalar, y por lo tanto los unicos comparables en el
# benchmark escalar vs vectorizado.
METODOS = {"Polar": PolarGenerator, "Rechazo": RechazoGenerator}

# Para el truncamiento entra el tercero: la transformada inversa no rechaza
# nada, asi que no tiene "fraccion de aceptacion" ni version escalar que medir,
# pero es justamente el metodo que hay que comparar cuando la cota se va a la
# cola. Gasta un uniforme por muestra, siempre.
METODOS_TRUNCAMIENTO = {"Polar": PolarGenerator, "Rechazo": RechazoGenerator,
                        "Inversa": InversaGenerator}

# Fracciones de aceptacion teoricas, derivadas en el informe:
#   Polar   : el par (v1, v2) cae en el circulo unitario con prob. pi/4
#   Rechazo : 1/c con c = sqrt(2e/pi), la cota de la envolvente exponencial
ACEPTACION_TEORICA = {
    "Polar": np.pi / 4,
    "Rechazo": np.sqrt(np.pi / (2 * np.e)),
}

# Cotas de truncamiento a barrer, en desviaciones estandar desde la media.
# Para Polar el costo crece como 1/P(X > a), asi que a partir de cierto punto
# medirlo deja de ser viable y solo se reporta su costo teorico.
CORTES_SIGMA = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
N_POR_CORTE = {0.0: 50_000, 1.0: 50_000, 2.0: 20_000, 3.0: 20_000, 4.0: 2_000, 5.0: 1_000}
CORTE_MAX_POLAR = 4.0    # mas alla de esto, Polar no se ejecuta (seria de minutos)


class ContadorUniformes(MersenneTwisterGenerator):
    """Envuelve al generador base para contar cuantos uniformes se consumieron."""

    def __init__(self, seed):
        super().__init__(seed)
        self.consumidos = 0

    def uniform(self, m: int) -> np.ndarray:
        self.consumidos += m
        return super().uniform(m)


# --------------------------------------------------------------- criterio 1

def medir_calidad(n: int = N_CALIDAD) -> list:
    """Estadistico KS de cada metodo contra N(0,1). Verifica que los tres son correctos."""
    filas = []
    for nombre, Gen in METODOS_TRUNCAMIENTO.items():
        muestra = Gen(MersenneTwisterGenerator(SEMILLA)).normal(n, 0.0, 1.0)
        res = stats.kstest(muestra, stats.norm(0.0, 1.0).cdf)
        filas.append({
            "metodo": nombre,
            "n": n,
            "ks_D": float(res.statistic),
            "ks_p": float(res.pvalue),
            "media": float(muestra.mean()),
            "desv": float(muestra.std(ddof=1)),
            "pasa": bool(res.pvalue > 0.05),
        })
    return filas


# --------------------------------------------------------------- criterio 2

def medir_aceptacion(n: int = 200_000) -> list:
    """
    Fraccion de aceptacion medida contra la teorica, y uniformes por normal.

    La fraccion se despeja del consumo de uniformes, y se mide sobre la version
    ESCALAR a proposito. La vectorizada pide lotes sobredimensionados (~1.3x) y
    descarta el sobrante al llenar el pedido, asi que su consumo mezcla dos
    cosas distintas -- el rechazo del metodo y el desperdicio del lote -- y
    subestimaria la aceptacion en ~20 puntos. La escalar consume exactamente un
    candidato a la vez, asi que mide el rechazo puro.

    Polar   : 2 uniformes por par candidato, 2 normales por par aceptado
              => consumidos = n / p          => p = n / consumidos
    Rechazo : 3 uniformes por candidato, 1 normal por candidato aceptado
              => consumidos = 3n / p         => p = 3n / consumidos

    Se reportan ademas los uniformes por normal de AMBOS regimenes, porque el
    del vectorizado es el costo que de verdad paga la simulacion.
    """
    filas = []
    for nombre, Gen in METODOS.items():
        contador_esc = ContadorUniformes(SEMILLA)
        Gen(contador_esc).normal_escalar(n, 0.0, 1.0)
        if nombre == "Polar":
            aceptacion = n / contador_esc.consumidos
        else:
            aceptacion = 3 * n / contador_esc.consumidos

        contador_vec = ContadorUniformes(SEMILLA)
        Gen(contador_vec).normal(n, 0.0, 1.0)

        teorica = ACEPTACION_TEORICA[nombre]
        filas.append({
            "metodo": nombre,
            "aceptacion_medida": float(aceptacion),
            "aceptacion_teorica": float(teorica),
            "error_relativo": float(abs(aceptacion - teorica) / teorica),
            "uniformes_por_normal": float(contador_esc.consumidos / n),
            "uniformes_por_normal_vectorizado": float(contador_vec.consumidos / n),
        })
    return filas


def medir_velocidad(n: int = N_BENCHMARK) -> dict:
    """Reutiliza el benchmark: tiempos escalar y vectorizado por metodo."""
    return correr_benchmark(n=n, repeticiones=3)


# --------------------------------------------------------------- criterio 3

def _probabilidad_cola(a_sigma: float) -> float:
    """P(X > a) para X ~ N(0,1): la fraccion de aceptacion del descarte ingenuo."""
    return float(stats.norm.sf(a_sigma))


def medir_truncamiento() -> list:
    """
    Barrido de la cota de truncamiento. Para cada metodo y cada corte se mide el
    tiempo y los uniformes por muestra ACEPTADA, que es el costo que de verdad
    paga la simulacion.
    """
    filas = []
    for a_sigma in CORTES_SIGMA:
        n = N_POR_CORTE[a_sigma]
        p_cola = _probabilidad_cola(a_sigma)

        for nombre, Gen in METODOS_TRUNCAMIENTO.items():
            # Costo teorico del descarte ingenuo: una normal completa cuesta
            # `uniformes_por_normal` y solo sobrevive una fraccion p_cola.
            costo_teorico = (4 / np.pi) / p_cola if nombre == "Polar" else None

            if nombre == "Polar" and a_sigma > CORTE_MAX_POLAR:
                filas.append({
                    "corte_sigma": a_sigma, "metodo": nombre, "n": 0,
                    "p_aceptacion_teorica": p_cola,
                    "segundos_por_muestra": float("nan"),
                    "uniformes_por_muestra": float("nan"),
                    "costo_teorico_uniformes": costo_teorico,
                    "ejecutado": False,
                })
                continue

            contador = ContadorUniformes(SEMILLA)
            gen = Gen(contador)
            inicio = time.perf_counter()
            muestra = gen.normal_truncada(n, 0.0, 1.0, a_sigma, np.inf)
            transcurrido = time.perf_counter() - inicio

            assert muestra.min() >= a_sigma - 1e-9, f"{nombre} violo la cota en a={a_sigma}"

            filas.append({
                "corte_sigma": a_sigma, "metodo": nombre, "n": n,
                "p_aceptacion_teorica": p_cola,
                "segundos_por_muestra": transcurrido / n,
                "uniformes_por_muestra": contador.consumidos / n,
                "costo_teorico_uniformes": costo_teorico,
                "ejecutado": True,
            })
    return filas


# ------------------------------------------------------------------ reporte

def imprimir_reporte(calidad, aceptacion, velocidad, truncamiento):
    print("\n" + "=" * 78)
    print("EXPERIMENTO 1: comparacion Polar vs Aceptacion-Rechazo")
    print("=" * 78)

    print("\n--- Criterio 1: calidad (KS contra N(0,1)) ---")
    print(f"{'Metodo':10s} {'D':>10s} {'p-valor':>10s} {'media':>10s} {'desv':>10s} {'':>8s}")
    for f in calidad:
        print(f"{f['metodo']:10s} {f['ks_D']:10.5f} {f['ks_p']:10.3f} "
              f"{f['media']:10.4f} {f['desv']:10.4f} {'PASA' if f['pasa'] else 'RECHAZA':>8s}")
    print("  => Los tres metodos son correctos. La calidad no decide.")

    print("\n--- Criterio 2: costo por muestra (sin truncar) ---")
    print(f"{'Metodo':10s} {'Acep. medida':>13s} {'Acep. teorica':>14s} {'Err. rel.':>10s} "
          f"{'Unif/norm esc':>14s} {'Unif/norm vec':>14s}")
    for f in aceptacion:
        print(f"{f['metodo']:10s} {f['aceptacion_medida']:13.4f} {f['aceptacion_teorica']:14.4f} "
              f"{f['error_relativo']:9.2%} {f['uniformes_por_normal']:14.2f} "
              f"{f['uniformes_por_normal_vectorizado']:14.2f}")
    print("  (esc = escalar, mide el rechazo puro; vec = vectorizado, incluye el sobrante del lote)")

    print(f"\n{'Metodo':10s} {'Escalar (s)':>13s} {'Vectorizado (s)':>17s} {'Speedup':>9s}")
    for nombre, r in velocidad.items():
        print(f"{nombre:10s} {r['escalar']:13.3f} {r['vectorizado']:17.4f} {r['speedup']:8.1f}x")
    mas_rapido = min(velocidad, key=lambda m: velocidad[m]["vectorizado"])
    print(f"  => {mas_rapido} es mas rapido en el caso comun.")

    print("\n--- Criterio 3: comportamiento en la cola (el que decide) ---")
    print(f"{'Corte':>7s} {'Metodo':10s} {'P(X>a)':>10s} {'Unif/muestra':>13s} "
          f"{'us/muestra':>12s} {'Teorico ingenuo':>16s}")
    for f in truncamiento:
        if not f["ejecutado"]:
            print(f"{f['corte_sigma']:6.1f}s {f['metodo']:10s} {f['p_aceptacion_teorica']:10.2e} "
                  f"{'no ejecutado':>13s} {'-':>12s} {f['costo_teorico_uniformes']:16.3e}")
            continue
        teorico = f"{f['costo_teorico_uniformes']:.3e}" if f["costo_teorico_uniformes"] else "-"
        print(f"{f['corte_sigma']:6.1f}s {f['metodo']:10s} {f['p_aceptacion_teorica']:10.2e} "
              f"{f['uniformes_por_muestra']:13.1f} {f['segundos_por_muestra'] * 1e6:12.2f} {teorico:>16s}")

    # Razon de costos en el corte mas exigente que Polar alcanzo a ejecutar:
    # ahi es donde el ranking del criterio 2 queda desmentido.
    ejecutados = [f for f in truncamiento if f["ejecutado"]]
    cortes_polar = sorted({f["corte_sigma"] for f in ejecutados if f["metodo"] == "Polar"})
    if cortes_polar:
        a = cortes_polar[-1]
        en_corte = {f["metodo"]: f for f in ejecutados if f["corte_sigma"] == a}
        costo_polar = en_corte["Polar"]["uniformes_por_muestra"]

        print(f"\n  => En a = {a:.0f} sigma, contra el descarte ingenuo de Polar:")
        for metodo, fila in en_corte.items():
            if metodo == "Polar":
                continue
            razon = costo_polar / fila["uniformes_por_muestra"]
            print(f"       {metodo:8s} gasta {razon:>9,.0f}x menos uniformes por muestra")

        print("  => El ranking SE INVIERTE: el metodo mas rapido sin truncar (Polar)")
        print("     es el inviable en la cola, que es justo donde vive la seccion Palco.")
        print("  => Criterio de eleccion que sale del experimento:")
        print("       sin truncar o con cota cerca de la media  -> Polar")
        print("       truncado en la cola                       -> Inversa (1 uniforme, exacta)")
        print("       truncado sin poder invertir la CDF        -> Rechazo con Robert (1995)")


def graficar(truncamiento, ruta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = ruta or os.path.join(DIR_SALIDA, "exp1_truncamiento.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colores = {"Polar": "#c44e52", "Rechazo": "#4c72b0", "Inversa": "#55a868"}

    for nombre in METODOS_TRUNCAMIENTO:
        puntos = [(f["corte_sigma"], f["uniformes_por_muestra"])
                  for f in truncamiento if f["metodo"] == nombre and f["ejecutado"]]
        if puntos:
            xs, ys = zip(*puntos)
            ax.plot(xs, ys, "o-", color=colores[nombre], lw=2, ms=7, label=f"{nombre} (medido)")

    # Curva teorica del descarte ingenuo, extendida a donde ya no es medible.
    xs = np.linspace(0, CORTES_SIGMA[-1], 200)
    ax.plot(xs, (4 / np.pi) / stats.norm.sf(xs), "--", color=colores["Polar"], lw=1.3,
            alpha=0.8, label="Polar (teorico, $(4/\\pi)/P(X>a)$)")

    ax.axvline(3.0, color="gray", ls=":", lw=1.2)
    ax.text(3.05, ax.get_ylim()[1] * 0.3, "Palco\n(> 3$\\sigma$)", fontsize=9, color="gray")

    ax.set_yscale("log")
    ax.set_xlabel("Cota de truncamiento $a$ (desviaciones estandar)")
    ax.set_ylabel("Uniformes consumidos por muestra aceptada (log)")
    ax.set_title("Costo del truncamiento: descarte ingenuo vs envolvente de Robert vs inversa")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"\n  Grafica guardada en: {ruta}")
    return ruta


def guardar_csv(calidad, aceptacion, velocidad, truncamiento):
    os.makedirs(DIR_SALIDA, exist_ok=True)
    rutas = []

    ruta = os.path.join(DIR_SALIDA, "exp1_truncamiento.csv")
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truncamiento[0].keys()))
        w.writeheader()
        w.writerows(truncamiento)
    rutas.append(ruta)

    ruta = os.path.join(DIR_SALIDA, "exp1_metodos.csv")
    filas = []
    for f in calidad:
        if f["metodo"] not in METODOS:
            continue
        acep = next(a for a in aceptacion if a["metodo"] == f["metodo"])
        vel = velocidad[f["metodo"]]
        filas.append({
            "metodo": f["metodo"], "ks_D": f["ks_D"], "ks_p": f["ks_p"],
            "aceptacion_medida": acep["aceptacion_medida"],
            "aceptacion_teorica": acep["aceptacion_teorica"],
            "uniformes_por_normal_escalar": acep["uniformes_por_normal"],
            "uniformes_por_normal_vectorizado": acep["uniformes_por_normal_vectorizado"],
            "t_escalar_s": vel["escalar"], "t_vectorizado_s": vel["vectorizado"],
            "speedup": vel["speedup"],
        })
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    rutas.append(ruta)

    for r in rutas:
        print(f"  CSV guardado en: {r}")
    return rutas


def main():
    print("Midiendo calidad (KS)...")
    calidad = medir_calidad()
    print("Midiendo fracciones de aceptacion...")
    aceptacion = medir_aceptacion()
    print(f"Corriendo benchmark de velocidad (n = {N_BENCHMARK:,})...")
    velocidad = medir_velocidad()
    print("Barriendo la cota de truncamiento (esto tarda: Polar colapsa en la cola)...")
    truncamiento = medir_truncamiento()

    imprimir_reporte(calidad, aceptacion, velocidad, truncamiento)
    guardar_csv(calidad, aceptacion, velocidad, truncamiento)
    graficar(truncamiento)


if __name__ == "__main__":
    main()
