"""
Experimento 3: cual defensa anti-bot funciona, y a costa de que.

El experimento 2 mostro que el precio dinamico sube el ingreso pero no cambia
QUIEN compra: con o sin el, los bots se llevan el 100% del inventario. El
precio es una palanca de recaudacion, no de acceso. Este experimento ataca el
otro problema.

Se comparan las cuatro defensas de modelo/politicas.py contra la linea base
"sin defensa", con 200 replicas y Numeros Aleatorios Comunes: la misma semilla
produce las mismas llegadas, valuaciones, paciencias y tiempos de checkout bajo
todas las politicas, asi que la diferencia observada es atribuible a la
politica y no al ruido. Los intervalos de confianza son PAREADOS contra la
linea base por esa misma razon.

La metrica principal NO es el ingreso. Es la fraccion de boletos que termina en
manos de humanos, porque ese es el problema que una boleteria bajo ataque tiene
que resolver. El ingreso, los abandonos por impaciencia y los humanos
bloqueados por error se reportan como el COSTO de cada defensa: casi todas
compran acceso a cambio de algo.

Ademas se barre el intercambio del captcha (tasa de deteccion contra tasa de
falso positivo), que es la decision de diseno real: un filtro que atrapa mas
bots casi siempre atrapa tambien mas humanos, y no es obvio donde conviene
pararse.

Uso:
    python -m experimentos.exp3_politicas
    python -m experimentos.exp3_politicas --replicas 50 --sin-barrido
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boleteria.modelo.politicas import Captcha, CATALOGO, crear_politica
from experimentos.exp2_precios import _correr_replica

N_REPLICAS = 200
SEMILLA_BASE = 5000      # 4 streams por replica, igual que el experimento 2
ETA = 0.10
DIR_SALIDA = "resultados"

# Barrido del intercambio del captcha. Son menos replicas por celda porque lo
# que interesa es la forma de la superficie, no el tercer decimal.
#
# La rejilla del falso positivo llega hasta 99% a proposito. En el rango
# razonable (1% a 20%) las tres columnas salen IDENTICAS, y con solo ese rango
# el resultado parece un error de programacion. No lo es: hay unas 130,000
# llegadas humanas peleando por 25,000 boletos, asi que bloquear al 20% de los
# humanos sigue dejando muchos mas humanos que inventario y no cambia quien
# compra. El falso positivo recien empieza a costar acceso arriba del 90%,
# cuando la oferta humana por fin cae por debajo del inventario. Hay que
# incluir esa zona para que la superficie muestre donde esta el codo.
DETECCIONES = [0.70, 0.90, 0.99]
FALSOS_POSITIVOS = [0.01, 0.20, 0.60, 0.90, 0.99]
REPLICAS_BARRIDO = 30

METRICAS = [
    ("fraccion_humanos", "Boletos a humanos", "{:.2%}"),
    ("ingreso", "Ingreso (Q)", "{:,.0f}"),
    ("abandonos_con_inventario", "Abandonos (con inv.)", "{:,.0f}"),
    ("bloqueados_bots", "Bots bloqueados", "{:,.0f}"),
    ("bloqueados_humanos", "Humanos bloq. (FP)", "{:,.0f}"),
]


def _metricas(resumen):
    """Extrae del resumen del motor las metricas que compara el experimento."""
    vendidos = resumen["boletos_vendidos"]
    return {
        "fraccion_humanos": resumen["boletos_humanos"] / vendidos if vendidos else 0.0,
        "boletos_humanos": float(resumen["boletos_humanos"]),
        "ingreso": float(resumen["ingreso"]),
        "abandonos_impaciencia": float(resumen["abandonos_impaciencia"]),
        "abandonos_con_inventario": float(resumen["abandonos_con_inventario"]),
        "bloqueados_por_politica": float(resumen["bloqueados_por_politica"]),
        "bloqueados_bots": float(resumen["bloqueados_bots"]),
        "bloqueados_humanos": float(resumen["bloqueados_humanos"]),
        "precio_promedio": float(resumen["precio_promedio"]),
    }


def correr_experimento(n_replicas=N_REPLICAS, politicas=None):
    """Para cada replica corre TODAS las politicas con la misma semilla (CRN)."""
    politicas = politicas or list(CATALOGO)
    filas = []
    for k in range(n_replicas):
        seed = SEMILLA_BASE + 4 * k
        for nombre in politicas:
            resumen = _correr_replica(seed, eta=ETA, politica=crear_politica(nombre))
            fila = {"replica": k, "semilla": seed, "politica": nombre}
            fila.update(_metricas(resumen))
            filas.append(fila)
        if (k + 1) % 10 == 0:
            print(f"  ... {k + 1}/{n_replicas} replicas")
    return filas


def _ic_pareado(diferencias, conf=0.95):
    d = np.asarray(diferencias, dtype=float)
    n = d.size
    media = d.mean()
    if n < 2 or d.std(ddof=1) == 0:
        return media, media, media
    error = d.std(ddof=1) / np.sqrt(n)
    t_critico = stats.t.ppf(0.5 + conf / 2, df=n - 1)
    return media, media - t_critico * error, media + t_critico * error


def _por_politica(filas, politica, clave):
    return np.array([f[clave] for f in filas if f["politica"] == politica])


def analizar(filas, base="ninguna"):
    politicas = list(dict.fromkeys(f["politica"] for f in filas))
    print("\n" + "=" * 96)
    print(f"EXPERIMENTO 3: politicas anti-bot ({len(_por_politica(filas, base, 'ingreso'))} "
          f"replicas, CRN, IC 95% pareado contra '{base}')")
    print("=" * 96)

    encabezado = f"{'Politica':16s}"
    for _, etiqueta, _ in METRICAS:
        encabezado += f"{etiqueta:>20s}"
    print("\n" + encabezado)
    print("-" * len(encabezado))

    resumen = {}
    for nombre in politicas:
        linea = f"{nombre:16s}"
        resumen[nombre] = {}
        for clave, _, formato in METRICAS:
            valores = _por_politica(filas, nombre, clave)
            resumen[nombre][clave] = float(valores.mean())
            linea += f"{formato.format(valores.mean()):>20s}"
        print(linea)

    print(f"\nGanancia en acceso contra '{base}' (puntos porcentuales de boletos a humanos):")
    print(f"{'Politica':16s} {'Ganancia':>12s} {'IC 95%':>26s} {'Significativa':>14s}")
    print("-" * 72)
    base_frac = _por_politica(filas, base, "fraccion_humanos")
    for nombre in politicas:
        if nombre == base:
            continue
        dif = (_por_politica(filas, nombre, "fraccion_humanos") - base_frac) * 100
        media, lo, hi = _ic_pareado(dif)
        significativa = "SI" if lo > 0 or hi < 0 else "NO"
        print(f"{nombre:16s} {media:+11.2f}pp [{lo:+10.2f}, {hi:+10.2f}] {significativa:>14s}")

    print(f"\nCosto de cada defensa contra '{base}':")
    print(f"{'Politica':16s} {'Ingreso':>16s} {'Abandonos extra':>18s} "
          f"{'Humanos bloq. (FP)':>20s}")
    print("-" * 74)
    base_ing = _por_politica(filas, base, "ingreso")
    base_aba = _por_politica(filas, base, "abandonos_con_inventario")
    for nombre in politicas:
        if nombre == base:
            continue
        d_ing, _, _ = _ic_pareado(_por_politica(filas, nombre, "ingreso") - base_ing)
        d_aba, _, _ = _ic_pareado(_por_politica(filas, nombre, "abandonos_con_inventario") - base_aba)
        bloq = _por_politica(filas, nombre, "bloqueados_humanos").mean()
        print(f"{nombre:16s} {d_ing:+16,.0f} {d_aba:+18,.0f} {bloq:20,.0f}")

    mejor = max((p for p in politicas if p != base),
                key=lambda p: resumen[p]["fraccion_humanos"])
    print(f"\n  => Mejor acceso: '{mejor}' con "
          f"{resumen[mejor]['fraccion_humanos']:.1%} de boletos a humanos "
          f"(base: {resumen[base]['fraccion_humanos']:.1%}).")
    abandonos_extra = (resumen[mejor]['abandonos_con_inventario']
                       - resumen[base]['abandonos_con_inventario'])
    print(f"  => Le cuesta Q {resumen[base]['ingreso'] - resumen[mejor]['ingreso']:,.0f} "
          f"de ingreso y {abandonos_extra:,.0f} abandonos adicionales "
          f"(contando solo los que ocurren con inventario disponible).")
    return resumen


def barrido_captcha(n_replicas=REPLICAS_BARRIDO):
    """
    Barre el intercambio deteccion / falso positivo del captcha.

    Cada celda es una configuracion del filtro; se reporta la fraccion de
    boletos que llega a humanos. La pregunta que responde: subir la deteccion
    del 90% al 99% compensa si eso obliga a subir el falso positivo?

    La respuesta, adelantada: en este modelo la deteccion lo es todo y el falso
    positivo casi no cuesta acceso hasta niveles absurdos, porque los humanos
    estan en exceso de oferta frente al inventario. Lo que si cuesta el falso
    positivo es experiencia de usuario -- los humanos bloqueados por error se
    reportan aparte en la tabla principal.
    """
    print(f"\nBarriendo el captcha ({len(DETECCIONES)}x{len(FALSOS_POSITIVOS)} "
          f"configuraciones x {n_replicas} replicas)...")
    malla = np.zeros((len(DETECCIONES), len(FALSOS_POSITIVOS)))
    filas = []

    for i, deteccion in enumerate(DETECCIONES):
        for j, falso_positivo in enumerate(FALSOS_POSITIVOS):
            fracciones = []
            for k in range(n_replicas):
                politica = Captcha(deteccion=deteccion, falso_positivo=falso_positivo)
                resumen = _correr_replica(SEMILLA_BASE + 4 * k, eta=ETA, politica=politica)
                fracciones.append(_metricas(resumen)["fraccion_humanos"])
            malla[i, j] = float(np.mean(fracciones))
            filas.append({"deteccion": deteccion, "falso_positivo": falso_positivo,
                          "fraccion_humanos": malla[i, j],
                          "ee": float(np.std(fracciones, ddof=1) / np.sqrt(n_replicas))})
        print(f"  deteccion {deteccion:.0%} lista")
    return malla, filas


def graficar(filas, malla=None, base="ninguna"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(DIR_SALIDA, exist_ok=True)
    politicas = list(dict.fromkeys(f["politica"] for f in filas))

    ancho = 2 if malla is not None else 1
    fig, axes = plt.subplots(1, ancho, figsize=(7.5 * ancho, 5))
    ax1 = axes[0] if ancho > 1 else axes

    medias, errores = [], []
    for nombre in politicas:
        v = _por_politica(filas, nombre, "fraccion_humanos") * 100
        medias.append(v.mean())
        errores.append(1.96 * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)

    colores = ["#c44e52" if p == base else "#4c72b0" for p in politicas]
    barras = ax1.bar(politicas, medias, yerr=errores, capsize=5, color=colores)
    ax1.bar_label(barras, fmt="%.1f%%", padding=3, fontsize=9)
    ax1.set_ylabel("Boletos que llegan a humanos (%)")
    ax1.set_title("Eficacia de cada defensa (IC 95%)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.25)

    if malla is not None:
        ax2 = axes[1]
        im = ax2.imshow(malla * 100, cmap="viridis", aspect="auto", origin="lower")
        ax2.set_xticks(range(len(FALSOS_POSITIVOS)))
        ax2.set_xticklabels([f"{v:.0%}" for v in FALSOS_POSITIVOS])
        ax2.set_yticks(range(len(DETECCIONES)))
        ax2.set_yticklabels([f"{v:.0%}" for v in DETECCIONES])
        ax2.set_xlabel("Falso positivo (humanos bloqueados por error)")
        ax2.set_ylabel("Deteccion (bots bloqueados)")
        ax2.set_title("Intercambio del captcha:\n% de boletos que llega a humanos")
        for i in range(malla.shape[0]):
            for j in range(malla.shape[1]):
                ax2.text(j, i, f"{malla[i, j] * 100:.1f}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax2, label="% a humanos")

    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDA, "exp3_politicas.png")
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  Grafica guardada en: {ruta}")
    return ruta


def guardar_csv(filas, filas_barrido=None):
    os.makedirs(DIR_SALIDA, exist_ok=True)
    ruta = os.path.join(DIR_SALIDA, "exp3_politicas.csv")
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"  CSV guardado en: {ruta}")

    if filas_barrido:
        ruta2 = os.path.join(DIR_SALIDA, "exp3_barrido_captcha.csv")
        with open(ruta2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_barrido[0].keys()))
            w.writeheader()
            w.writerows(filas_barrido)
        print(f"  CSV guardado en: {ruta2}")


def main():
    parser = argparse.ArgumentParser(description="Experimento 3: politicas anti-bot")
    parser.add_argument("--replicas", type=int, default=N_REPLICAS)
    parser.add_argument("--sin-barrido", action="store_true",
                        help="omite el barrido del captcha (es la parte lenta)")
    args = parser.parse_args()

    print(f"Corriendo {args.replicas} replicas x {len(CATALOGO)} politicas (CRN)...")
    filas = correr_experimento(args.replicas)
    analizar(filas)

    malla, filas_barrido = (None, None)
    if not args.sin_barrido:
        malla, filas_barrido = barrido_captcha()

    guardar_csv(filas, filas_barrido)
    graficar(filas, malla)


if __name__ == "__main__":
    main()
