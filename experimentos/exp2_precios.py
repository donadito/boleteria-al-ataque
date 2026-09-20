"""
Experimento 2: precio fijo vs precio dinamico con 200 replicas y Numeros
Aleatorios Comunes (CRN).

Cada replica k fija una semilla estatica y corre la MISMA simulacion bajo dos
politicas de precio -- fija (eta = 0, el precio nunca se mueve) y dinamica
(eta = 0.10) -- de modo que ambas comparten exactamente las mismas llegadas,
valuaciones, paciencias, tiempos de checkout, rafagas de bots y cantidades de
boletos. Solo difiere la politica de precio. Al restar los ingresos de la misma
semilla (comparacion pareada), la varianza comun se cancela y queda "pura" la
diferencia atribuible a la politica, tal como pide el Frente 3.

Los flujos aleatorios se separan en cuatro streams derivados de la semilla para
que el desfase de un stream (los pagos, cuyo numero de extracciones depende de
cuantas compras ocurren) no contamine a los demas:
    seed     -> variables normales (valuacion, paciencia, checkout, bots)
    seed + 1 -> proceso de llegadas NHPP
    seed + 2 -> uniformes de generacion (cantidad de boletos, desfase de bots)
    seed + 3 -> uniformes de pago dentro del motor

NOTA: el motor todavia no cablea correr_fase (Frente 2), asi que este harness
maneja el avance temporal por minutos usando los metodos publicos ya existentes
(procesar_cola_espera, ejecutar_decision_compra via la cola, y el controlador de
precios). Cuando correr_fase quede listo, basta reemplazar _correr_replica por
una llamada a motor.correr_simulacion_completa.

Uso:
    python -m experimentos.exp2_precios
    python experimentos/exp2_precios.py
"""
import os
import sys
import csv

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boleteria.rng.lcg import MersenneTwisterGenerator
from boleteria.rng.polar import PolarGenerator
from boleteria.rng.poisson import LlegadasNHPP
from boleteria.modelo.calendario import DURACION_FASE, obtener_fases
from boleteria.modelo.entidades import crear_secciones
from boleteria.modelo.generacion import crear_flota_bots, generar_llegadas
from boleteria.modelo.motor import MotorSimulacion
from boleteria.modelo.precios import ControladorPrecios

N_REPLICAS = 200
SEMILLA_BASE = 1000          # semillas estaticas: SEMILLA_BASE + 4*k por replica
ETA_DINAMICO = 0.10
DIR_SALIDA = "resultados"

# Cada cuanto (en minutos de simulacion) el controlador recalcula el precio.
# Es la velocidad de reaccion del sistema de precios dinamico. Es CRITICO para el
# experimento: los 22,000 bots vacian el inventario en la rafaga de ~5 s inicial,
# asi que con un paso de 1 minuto el precio no alcanza a reaccionar y el precio
# dinamico produce el MISMO ingreso que el fijo (diferencia 0). Con un paso mas
# fino (1 segundo) el precio sube durante la rafaga y el dinamico captura mas
# ingreso. Se deja como parametro para que el grupo lo afine junto con los topes
# de precio (Frente 2) hasta aterrizar en el optimo analitico de ~18.47 M.
UN_SEGUNDO = 1.0 / 60.0
PASO_PRECIO = UN_SEGUNDO


def _liberar_inventario(secciones, liberacion):
    """Agrega a las secciones persistentes el inventario que libera la fase."""
    for sec, delta in zip(secciones, crear_secciones(liberacion)):
        sec.inventario += delta.inventario


def _procesar_fase(motor, controlador, secciones, llegadas, fase, paso=PASO_PRECIO):
    """
    Avanza la fase en pasos de `paso` minutos: encola las llegadas del paso, drena
    la cola con la politica FIFO del motor y actualiza los precios en cada ventana.
    El tamano de `paso` fija la velocidad de reaccion del precio dinamico.
    """
    # Cada fase es un evento de venta nuevo: cola y slots de checkout frescos.
    motor.cola_espera = []
    motor.relojes_checkout = []
    motor.inventario_global = sum(sec.inventario for sec in secciones)

    llegadas.sort(key=lambda e: e.t_llegada)
    i, n = 0, len(llegadas)
    t = fase.hora_apertura
    fin = fase.hora_apertura + DURACION_FASE

    while t < fin:
        while i < n and llegadas[i].t_llegada < t + paso:
            motor.cola_espera.append(llegadas[i])
            i += 1

        motor.procesar_cola_espera(t, secciones)
        controlador.actualizar_precios(secciones, fase.liberacion_inventario)

        # Sin inventario ya no hay ventas posibles; sin llegadas ni cola, terminamos.
        if motor.inventario_global <= 0:
            break
        if i >= n and not motor.cola_espera:
            break
        t += paso


def _correr_replica(seed, eta, paso=PASO_PRECIO):
    """Corre la simulacion completa (3 fases) con una semilla y una politica de precio."""
    gen_normal = PolarGenerator(MersenneTwisterGenerator(seed))
    nhpp = LlegadasNHPP(MersenneTwisterGenerator(seed + 1))
    gen_generacion = MersenneTwisterGenerator(seed + 2)   # boletos por compra + bots
    gen_pago = MersenneTwisterGenerator(seed + 3)         # pagos dentro del motor

    motor = MotorSimulacion(gen_normal, gen_pago)
    controlador = ControladorPrecios(eta=eta)

    fases = obtener_fases()
    secciones = crear_secciones(0)  # inventario inicial 0; cada fase lo va liberando
    total_bots = max(fase.bots_admitidos for fase in fases)
    flota = crear_flota_bots(total_bots, gen_normal)

    for fase in fases:
        _liberar_inventario(secciones, fase.liberacion_inventario)
        llegadas = generar_llegadas(fase, flota, nhpp, gen_normal, gen_generacion)
        _procesar_fase(motor, controlador, secciones, llegadas, fase, paso)

    return dict(motor.estadisticas)


def correr_experimento(n_replicas=N_REPLICAS):
    filas = []
    for k in range(n_replicas):
        seed = SEMILLA_BASE + 4 * k  # 4 streams por replica, semillas estaticas
        est_fijo = _correr_replica(seed, eta=0.0)
        est_din = _correr_replica(seed, eta=ETA_DINAMICO)
        filas.append({
            "replica": k,
            "semilla": seed,
            "ingreso_fijo": est_fijo["ingreso"],
            "ingreso_dinamico": est_din["ingreso"],
            "diferencia": est_din["ingreso"] - est_fijo["ingreso"],
            "boletos_bots_fijo": est_fijo["boletos_bots"],
            "boletos_bots_dinamico": est_din["boletos_bots"],
        })
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{n_replicas} replicas")
    return filas


def _ic_pareado(diferencias, conf=0.95):
    d = np.asarray(diferencias, dtype=float)
    n = d.size
    media = d.mean()
    error = d.std(ddof=1) / np.sqrt(n)
    t_critico = stats.t.ppf(0.5 + conf / 2, df=n - 1)
    return media, media - t_critico * error, media + t_critico * error


def analizar(filas):
    fijo = np.array([f["ingreso_fijo"] for f in filas])
    din = np.array([f["ingreso_dinamico"] for f in filas])
    dif = np.array([f["diferencia"] for f in filas])

    media_d, ic_lo, ic_hi = _ic_pareado(dif)

    # Reduccion de varianza que aporta CRN: comparado con estimar la diferencia
    # con corridas independientes (donde Var(D) = Var(fijo) + Var(din)).
    var_crn = dif.var(ddof=1)
    var_indep = fijo.var(ddof=1) + din.var(ddof=1)

    print("\n=== Experimento 2: precio fijo vs dinamico (200 replicas, CRN) ===\n")
    print(f"  Ingreso medio precio FIJO     : {fijo.mean():14,.0f}  (std {fijo.std(ddof=1):,.0f})")
    print(f"  Ingreso medio precio DINAMICO : {din.mean():14,.0f}  (std {din.std(ddof=1):,.0f})")
    print(f"  Ganancia media del dinamico   : {media_d:14,.0f}")
    print(f"  IC 95% pareado de la ganancia : [{ic_lo:,.0f} , {ic_hi:,.0f}]")
    sig = "SI" if ic_lo > 0 or ic_hi < 0 else "NO"
    print(f"  Diferencia significativa (IC no contiene 0): {sig}")
    print("\n  --- Reduccion de varianza por Numeros Aleatorios Comunes ---")
    print(f"  Var(D) con CRN (pareado)      : {var_crn:14,.0f}")
    print(f"  Var(D) si fueran independientes: {var_indep:14,.0f}")
    if var_crn > 0:
        print(f"  Factor de reduccion            : {var_indep / var_crn:14.1f}x")

    return {
        "ingreso_fijo_medio": float(fijo.mean()),
        "ingreso_dinamico_medio": float(din.mean()),
        "ganancia_media": float(media_d),
        "ic95": (float(ic_lo), float(ic_hi)),
        "var_crn": float(var_crn),
        "var_indep": float(var_indep),
    }


def guardar_csv(filas, ruta=None):
    ruta = ruta or os.path.join(DIR_SALIDA, "exp2_replicas.csv")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    print(f"\n  CSV guardado en: {ruta}")
    return ruta


def graficar(filas, resumen, ruta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = ruta or os.path.join(DIR_SALIDA, "exp2_replicas.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    fijo = np.array([f["ingreso_fijo"] for f in filas])
    din = np.array([f["ingreso_dinamico"] for f in filas])
    dif = np.array([f["diferencia"] for f in filas])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Ingreso medio con barras de error (IC 95%).
    medias = [fijo.mean(), din.mean()]
    errs = [1.96 * fijo.std(ddof=1) / np.sqrt(fijo.size),
            1.96 * din.std(ddof=1) / np.sqrt(din.size)]
    ax1.bar(["Fijo", "Dinamico"], medias, yerr=errs, capsize=6,
            color=["#c44e52", "#4c72b0"])
    ax1.set_ylabel("Ingreso medio")
    ax1.set_title("Ingreso por politica (IC 95%)")

    # Distribucion de la diferencia pareada.
    ax2.hist(dif, bins=25, color="#8172b3", alpha=0.8, edgecolor="white")
    ax2.axvline(dif.mean(), color="k", lw=1.5, label=f"media = {dif.mean():,.0f}")
    ax2.axvline(0, color="r", ls="--", lw=1.2, label="cero")
    ax2.set_xlabel("Ingreso dinamico - Ingreso fijo (pareado)")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Ganancia pareada del precio dinamico")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  Grafica guardada en: {ruta}")
    return ruta


def main():
    print(f"Corriendo {N_REPLICAS} replicas (precio fijo vs dinamico, CRN)...")
    filas = correr_experimento(N_REPLICAS)
    resumen = analizar(filas)
    guardar_csv(filas)
    graficar(filas, resumen)


if __name__ == "__main__":
    main()
