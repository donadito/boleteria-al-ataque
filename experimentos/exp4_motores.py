"""
Experimento 4: cuanto detalle del modelo de atencion hace falta.

El motor procesa arreglos de tiempos en dos pasadas y representa el recurso
limitado: 500 cajas de checkout con su arreglo de horas de liberacion, la
espera que eso genera y el abandono por impaciencia. Representar el recurso
cuesta codigo y cuesta explicarlo, asi que vale la pena medir si aporta.

Se comparan tres versiones del mismo recorrido secuencial, quitando piezas:

    solo_llegada          ordena por tiempo de llegada y recorre
    llegada_y_checkout    ordena por llegada + checkout y recorre
    con_cajas             ademas representa las 500 cajas, la espera y la
                          impaciencia (el motor del proyecto)

Con Numeros Aleatorios Comunes: la misma semilla produce las mismas llegadas,
valuaciones, paciencias y tiempos de checkout en las tres, asi que cualquier
diferencia es atribuible al modelo de atencion y nada mas.

TODO se corre con PRECIO FIJO (eta = 0). No es una simplificacion opcional: sin
representar cuando termina cada checkout no hay ritmo de venta por ventana
contra el cual el controlador pueda ajustar, asi que el precio dinamico no es
representable en las dos versiones reducidas. Compararlas con precio dinamico
seria comparar modelos distintos y atribuirselo al orden de atencion.

El experimento responde dos preguntas separadas:

  1. ¿Cambian los NUMEROS? Se mide la fraccion de boletos que llega a humanos
     bajo cada politica, con intervalos pareados contra el modelo completo.

  2. ¿Cambia lo que se puede PREGUNTAR? Sin cajas no hay espera, asi que no hay
     abandonos por impaciencia, ni longitud de cola, ni tiempo de espera, ni
     precio dinamico. Eso no aparece como una diferencia en los numeros:
     aparece como preguntas que un modelo contesta y el otro no.

Uso:
    python -m experimentos.exp4_motores
    python -m experimentos.exp4_motores --replicas 20
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boleteria.modelo.motor_simple import MOTORES
from boleteria.modelo.politicas import CATALOGO, crear_politica
from experimentos.exp2_precios import _correr_replica

N_REPLICAS = 50
SEMILLA_BASE = 9000
DIR_SALIDA = "resultados"
REFERENCIA = "con_cajas"

# Metricas que solo el modelo con cajas puede producir, con la razon.
CAPACIDADES = [
    ("abandonos_con_inventario", "Abandonos con inventario",
     "requiere representar la espera"),
    ("espera_promedio", "Espera promedio (min)",
     "requiere saber cuando se libera cada caja"),
    ("espera_maxima", "Espera maxima (min)",
     "requiere representar la cola"),
]


def correr_experimento(n_replicas=N_REPLICAS):
    filas = []
    for k in range(n_replicas):
        seed = SEMILLA_BASE + 4 * k
        for pol in CATALOGO:
            for nombre_motor, clase in MOTORES.items():
                inicio = time.perf_counter()
                r = _correr_replica(seed, eta=0.0, politica=crear_politica(pol),
                                    motor_clase=clase)
                transcurrido = time.perf_counter() - inicio
                vendidos = r["boletos_vendidos"]
                filas.append({
                    "replica": k, "semilla": seed, "politica": pol,
                    "motor": nombre_motor,
                    "fraccion_humanos": r["boletos_humanos"] / vendidos if vendidos else 0.0,
                    "boletos_vendidos": vendidos,
                    "boletos_humanos": r["boletos_humanos"],
                    "ingreso": r["ingreso"],
                    "abandonos_impaciencia": r["abandonos_impaciencia"],
                    "abandonos_con_inventario": r["abandonos_con_inventario"],
                    "espera_promedio": r["espera_promedio"],
                    "espera_maxima": r["espera_maxima"],
                    "segundos": transcurrido,
                })
        if (k + 1) % 10 == 0:
            print(f"  ... {k + 1}/{n_replicas} replicas")
    return filas


def _sel(filas, politica, motor, clave):
    return np.array([f[clave] for f in filas
                     if f["politica"] == politica and f["motor"] == motor], dtype=float)


def _ic_pareado(dif, conf=0.95):
    d = np.asarray(dif, dtype=float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return d.mean(), d.mean(), d.mean()
    error = d.std(ddof=1) / np.sqrt(d.size)
    t = stats.t.ppf(0.5 + conf / 2, df=d.size - 1)
    return d.mean(), d.mean() - t * error, d.mean() + t * error


def analizar(filas):
    motores = list(MOTORES)
    politicas = list(CATALOGO)
    n = len({f["replica"] for f in filas})

    print("\n" + "=" * 92)
    print(f"EXPERIMENTO 4: cuanto detalle del modelo de atencion hace falta "
          f"({n} replicas, CRN, precio fijo)")
    print("=" * 92)

    print("\n--- Pregunta 1: ¿cambian los numeros? (% de boletos que llega a humanos) ---\n")
    encabezado = f"{'Politica':16s}" + "".join(f"{m:>26s}" for m in motores)
    print(encabezado)
    print("-" * len(encabezado))
    for pol in politicas:
        linea = f"{pol:16s}"
        for m in motores:
            linea += f"{_sel(filas, pol, m, 'fraccion_humanos').mean():25.1%} "
        print(linea)

    print(f"\n--- Diferencia contra '{REFERENCIA}' (puntos porcentuales, IC 95% pareado) ---\n")
    print(f"{'Politica':16s} {'Version':26s} {'Dif.':>9s} {'IC 95%':>22s} {'Importa':>9s}")
    print("-" * 86)
    maxima = 0.0
    for pol in politicas:
        base = _sel(filas, pol, REFERENCIA, "fraccion_humanos")
        for m in motores:
            if m == REFERENCIA:
                continue
            dif = (_sel(filas, pol, m, "fraccion_humanos") - base) * 100
            media, lo, hi = _ic_pareado(dif)
            maxima = max(maxima, abs(media))
            # "Importa" = el IC excluye cero Y la diferencia supera 1 punto
            # porcentual. Con CRN y 50 replicas casi cualquier diferencia sale
            # significativa; lo que interesa es si ademas es grande.
            importa = "SI" if (lo > 0 or hi < 0) and abs(media) >= 1.0 else "no"
            print(f"{pol:16s} {m:26s} {media:+8.2f}pp [{lo:+7.2f}, {hi:+7.2f}] {importa:>9s}")

    print(f"\n  => Divergencia maxima entre versiones: {maxima:.1f} puntos porcentuales.")

    print("\n--- Pregunta 2: ¿que puede contestar cada version? ---\n")
    print(f"{'Metrica':30s}" + "".join(f"{m:>26s}" for m in motores))
    print("-" * (30 + 26 * len(motores)))
    for clave, etiqueta, _ in CAPACIDADES:
        linea = f"{etiqueta:30s}"
        for m in motores:
            valores = np.concatenate([_sel(filas, p, m, clave) for p in politicas])
            linea += (f"{'-- no lo modela --':>26s}" if np.allclose(valores, 0.0)
                      else f"{valores.mean():25.2f} ")
        print(linea)
    linea = f"{'Precio dinamico':30s}"
    for m in motores:
        linea += f"{'si' if m == REFERENCIA else '-- no lo modela --':>26s}"
    print(linea)

    print("\n  Razon de cada ausencia:")
    for _, etiqueta, razon in CAPACIDADES:
        print(f"    {etiqueta:30s} {razon}")
    print(f"    {'Precio dinamico':30s} requiere saber cuando termina cada checkout")

    print("\n--- Costo de correr cada version ---\n")
    print(f"{'Version':26s} {'Segundos por replica':>22s} {'Sobrecosto':>12s}")
    print("-" * 62)
    base_t = np.array([f["segundos"] for f in filas if f["motor"] == REFERENCIA]).mean()
    for m in motores:
        t = np.array([f["segundos"] for f in filas if f["motor"] == m]).mean()
        print(f"{m:26s} {t:22.3f} {t / base_t:11.2f}x")

    return maxima


def graficar(filas, ruta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(DIR_SALIDA, exist_ok=True)
    motores = list(MOTORES)
    politicas = list(CATALOGO)
    colores = {"solo_llegada": "#c44e52", "llegada_y_checkout": "#dd8452",
               "con_cajas": "#4c72b0"}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(politicas))
    ancho = 0.26

    for i, m in enumerate(motores):
        medias = [_sel(filas, p, m, "fraccion_humanos").mean() * 100 for p in politicas]
        errs = [1.96 * _sel(filas, p, m, "fraccion_humanos").std(ddof=1) * 100
                / np.sqrt(len(_sel(filas, p, m, "fraccion_humanos"))) for p in politicas]
        barras = ax.bar(x + (i - 1) * ancho, medias, ancho, yerr=errs, capsize=3,
                        label=m, color=colores[m])
        ax.bar_label(barras, fmt="%.1f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(politicas, rotation=15)
    ax.set_ylabel("Boletos que llegan a humanos (%)")
    ax.set_title("¿Cuanto detalle del modelo de atencion hace falta?\n"
                 "Mismas semillas, mismas variables aleatorias, precio fijo",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    ruta = ruta or os.path.join(DIR_SALIDA, "exp4_motores.png")
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  Grafica guardada en: {ruta}")
    return ruta


def guardar_csv(filas):
    os.makedirs(DIR_SALIDA, exist_ok=True)
    ruta = os.path.join(DIR_SALIDA, "exp4_motores.csv")
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"  CSV guardado en: {ruta}")
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Experimento 4: sensibilidad al motor")
    parser.add_argument("--replicas", type=int, default=N_REPLICAS)
    args = parser.parse_args()

    print(f"Corriendo {args.replicas} replicas x {len(CATALOGO)} politicas "
          f"x {len(MOTORES)} motores (CRN, precio fijo)...")
    filas = correr_experimento(args.replicas)
    analizar(filas)
    guardar_csv(filas)
    graficar(filas)


if __name__ == "__main__":
    main()
