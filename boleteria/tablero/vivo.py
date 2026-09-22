"""
Tablero en vivo: la demostracion visual de la simulacion.

Es lo que se proyecta durante la presentacion. Muestra, minuto a minuto de
reloj simulado, cuatro paneles:

    1. Inventario restante por seccion   -> se ve el sold-out por seccion
    2. Precio vigente por seccion        -> se ve reaccionar al controlador
    3. Boletos acumulados, bots vs humanos -> se ve quien se lleva el evento
    4. Sala de espera y cajas ocupadas   -> se ve el cuello de botella

La simulacion se corre PRIMERO y se graba; la animacion reproduce la grabacion
despues. Es a proposito: una animacion que llama al motor dentro del callback
se traba si un cuadro tarda de mas, y en una presentacion en vivo eso es
justo lo que no puede pasar. Correr los tres dias toma menos de un segundo, asi
que no hay nada que ganar haciendolo al vuelo.

Uso:
    python -m boleteria.tablero.vivo              # ventana interactiva
    python -m boleteria.tablero.vivo --gif        # guarda resultados/tablero.gif
    python -m boleteria.tablero.vivo --politica cupo_estricto
"""
import argparse
import os

import numpy as np

from ..modelo.calendario import obtener_fases
from ..modelo.entidades import CATALOGO_SECCIONES, crear_secciones
from ..modelo.generacion import crear_flota_bots, generar_llegadas
from ..modelo.motor import UN_SEGUNDO, MotorSimulacion
from ..modelo.precios import ControladorPrecios
from ..modelo.politicas import crear_politica
from ..rng.lcg import crear_generador_base
from ..rng.poisson import LlegadasNHPP
from ..rng.polar import PolarGenerator

SEMILLA = 20240501
DIR_SALIDA = "resultados"
COLORES = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

# Cada cuantos minutos de reloj simulado se guarda un cuadro. El ataque dura
# unos 15 s, asi que con 0.05 min (3 s) se perderia entero: hay que grabar fino.
RESOLUCION = 0.2 * UN_SEGUNDO


class Grabadora:
    """
    Observador del motor. Se le pasa a correr_fase y guarda una foto del
    sistema cada `resolucion` minutos de reloj simulado.

    El motor la invoca en cada compra concretada, que en una fase son miles de
    veces; el filtro por tiempo es lo que mantiene la grabacion en un tamano
    razonable. La ocupacion de cajas y el largo de la sala de espera se
    consultan al motor, que los deriva de los arreglos de tiempos de la pasada
    de asignacion.
    """

    def __init__(self, resolucion=RESOLUCION):
        self.resolucion = resolucion
        self.t = []
        self.inventarios = []      # una fila por foto, una columna por seccion
        self.precios = []
        self.bots = []
        self.humanos = []
        self.cola = []
        self.cajas = []
        self.fases = []            # (t_apertura, nombre) de cada fase
        self._ultimo = None

    def marcar_fase(self, fase):
        self.fases.append((fase.hora_apertura, fase.nombre))
        self._ultimo = None        # fuerza una foto al abrir la fase

    def __call__(self, reloj, motor, secciones):
        if self._ultimo is not None and reloj - self._ultimo < self.resolucion:
            return
        self._ultimo = reloj

        self.t.append(reloj)
        self.inventarios.append([sec.inventario for sec in secciones])
        self.precios.append([sec.precio_vigente for sec in secciones])
        self.bots.append(motor.estadisticas['boletos_bots'])
        self.humanos.append(motor.estadisticas['boletos_humanos'])
        self.cola.append(motor.en_espera_en(reloj))
        self.cajas.append(motor.cajas_ocupadas_en(reloj))

    def como_arreglos(self):
        return {
            "t": np.array(self.t),
            "inventarios": np.array(self.inventarios),
            "precios": np.array(self.precios),
            "bots": np.array(self.bots),
            "humanos": np.array(self.humanos),
            "cola": np.array(self.cola),
            "cajas": np.array(self.cajas),
            "fases": self.fases,
        }


def correr_y_grabar(semilla=SEMILLA, eta=0.10, politica=None, resolucion=RESOLUCION):
    """Corre los tres dias de venta grabando el estado para la animacion."""
    gen_normal = PolarGenerator(crear_generador_base("mt", semilla))
    nhpp = LlegadasNHPP(crear_generador_base("mt", semilla + 1))
    gen_generacion = crear_generador_base("mt", semilla + 2)
    gen_pago = crear_generador_base("mt", semilla + 3)

    motor = MotorSimulacion(gen_normal, gen_pago, politica=politica)
    controlador = ControladorPrecios(eta=eta)
    grabadora = Grabadora(resolucion)

    fases = obtener_fases()
    secciones = crear_secciones(0)
    flota = crear_flota_bots(max(f.bots_admitidos for f in fases), gen_normal)

    from ..modelo.motor import liberar_inventario
    for fase in fases:
        liberar_inventario(secciones, fase.liberacion_inventario)
        grabadora.marcar_fase(fase)
        llegadas = generar_llegadas(fase, flota, nhpp, gen_normal, gen_generacion)
        motor.correr_fase(fase, secciones, controlador, llegadas, observador=grabadora)

    return grabadora.como_arreglos(), motor.resumen()


def _eje_tiempo(datos):
    """
    Linea de tiempo que empalma las tres fases, en segundos.

    Las fases abren con 24 h de diferencia pero la accion de cada una dura
    segundos, asi que en tiempo absoluto la animacion serian tres destellos
    separados por dos dias en blanco. Se toma el tiempo DENTRO de cada fase y
    se desplaza por lo que duraron las anteriores: el eje queda monotono, sin
    huecos, y las fronteras entre fases se marcan aparte.

    Devuelve (segundos, fronteras) con las fronteras en el mismo eje.
    """
    t = datos["t"]
    aperturas = np.array([apertura for apertura, _ in datos["fases"]])

    # A que fase pertenece cada foto.
    indice = np.clip(np.searchsorted(aperturas, t, side="right") - 1, 0, aperturas.size - 1)
    relativo = t - aperturas[indice]

    # Desplazamiento acumulado: cada fase arranca donde termino la anterior.
    duraciones = np.array([relativo[indice == k].max() if np.any(indice == k) else 0.0
                           for k in range(aperturas.size)])
    desplazamiento = np.concatenate([[0.0], np.cumsum(duraciones)[:-1]])

    segundos = (relativo + desplazamiento[indice]) * 60.0
    fronteras = np.cumsum(duraciones)[:-1] * 60.0
    return segundos, fronteras


def animar(datos, resumen, ruta_gif=None, intervalo=40):
    import matplotlib
    if ruta_gif:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    nombres = [nombre for nombre, *_ in CATALOGO_SECCIONES]
    segundos, fronteras = _eje_tiempo(datos)
    n = segundos.size

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.suptitle("Boleteria bajo ataque - venta de 25,000 localidades en 3 fases",
                 fontsize=14, y=0.98)

    ax_inv, ax_pre = axes[0]
    ax_acu, ax_col = axes[1]

    # --- panel 1: inventario por seccion
    lineas_inv = [ax_inv.plot([], [], color=COLORES[k], lw=2, label=nombres[k])[0]
                  for k in range(len(nombres))]
    ax_inv.set_xlim(0, segundos.max() * 1.02)
    ax_inv.set_ylim(0, datos["inventarios"].max() * 1.08)
    ax_inv.set_ylabel("Boletos disponibles")
    ax_inv.set_title("Inventario por seccion (el sold-out es por seccion)", fontsize=10)
    ax_inv.legend(fontsize=8, loc="upper right")
    ax_inv.grid(alpha=0.25)

    # --- panel 2: precio vigente
    lineas_pre = [ax_pre.plot([], [], color=COLORES[k], lw=2, label=nombres[k])[0]
                  for k in range(len(nombres))]
    ax_pre.set_xlim(0, segundos.max() * 1.02)
    ax_pre.set_ylim(0, datos["precios"].max() * 1.1)
    ax_pre.set_ylabel("Precio vigente (Q)")
    ax_pre.set_title("Motor de precios (topes en 0.7x y 3.0x del base)", fontsize=10)
    ax_pre.grid(alpha=0.25)

    # --- panel 3: acumulado bots vs humanos
    linea_bots, = ax_acu.plot([], [], color="#c44e52", lw=2.5, label="Bots")
    linea_hum, = ax_acu.plot([], [], color="#4c72b0", lw=2.5, label="Humanos")
    ax_acu.set_xlim(0, segundos.max() * 1.02)
    ax_acu.set_ylim(0, max(datos["bots"].max(), datos["humanos"].max()) * 1.1 + 1)
    ax_acu.set_xlabel("Segundos de venta activa (las 3 fases empalmadas)")
    ax_acu.set_ylabel("Boletos comprados (acumulado)")
    ax_acu.set_title("Quien se lleva el evento", fontsize=10)
    ax_acu.legend(fontsize=9, loc="upper left")
    ax_acu.grid(alpha=0.25)

    # --- panel 4: sala de espera y cajas
    linea_cola, = ax_col.plot([], [], color="#8172b3", lw=2, label="Sala de espera")
    linea_caja, = ax_col.plot([], [], color="#55a868", lw=2, label="Cajas ocupadas (max 500)")
    ax_col.set_xlim(0, segundos.max() * 1.02)
    ax_col.set_ylim(0, max(datos["cola"].max(), 500) * 1.1)
    ax_col.set_xlabel("Segundos de venta activa (las 3 fases empalmadas)")
    ax_col.set_ylabel("Entidades")
    ax_col.set_title("Cuello de botella: 500 cajas concurrentes", fontsize=10)
    ax_col.legend(fontsize=8, loc="upper right")
    ax_col.grid(alpha=0.25)

    for ax in (ax_inv, ax_pre, ax_acu, ax_col):
        for frontera in fronteras:
            ax.axvline(frontera, color="gray", ls="--", lw=1.0, alpha=0.7)

    etiqueta = ax_inv.text(0.02, 0.06, "", transform=ax_inv.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Indice donde arranca cada fase, para rotular el cuadro en curso.
    inicios = [int(np.searchsorted(datos["t"], apertura)) for apertura, _ in datos["fases"]]

    def nombre_fase(i):
        actual = "?"
        for indice, (_, nombre) in zip(inicios, datos["fases"]):
            if i >= indice:
                actual = nombre
        return actual

    def init():
        for linea in lineas_inv + lineas_pre + [linea_bots, linea_hum, linea_cola, linea_caja]:
            linea.set_data([], [])
        etiqueta.set_text("")
        return lineas_inv + lineas_pre + [linea_bots, linea_hum, linea_cola, linea_caja, etiqueta]

    def actualizar(i):
        j = min(i + 1, n)
        x = segundos[:j]
        for k, linea in enumerate(lineas_inv):
            linea.set_data(x, datos["inventarios"][:j, k])
        for k, linea in enumerate(lineas_pre):
            linea.set_data(x, datos["precios"][:j, k])
        linea_bots.set_data(x, datos["bots"][:j])
        linea_hum.set_data(x, datos["humanos"][:j])
        linea_cola.set_data(x, datos["cola"][:j])
        linea_caja.set_data(x, datos["cajas"][:j])
        etiqueta.set_text(f"{nombre_fase(j - 1)}\nt = {segundos[j - 1]:6.1f} s")
        return lineas_inv + lineas_pre + [linea_bots, linea_hum, linea_cola, linea_caja, etiqueta]

    # Con miles de fotos la animacion seria eterna; se muestrea a ~300 cuadros.
    paso = max(1, n // 300)
    cuadros = list(range(0, n, paso)) + [n - 1]

    anim = FuncAnimation(fig, actualizar, frames=cuadros, init_func=init,
                         interval=intervalo, blit=False, repeat=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if ruta_gif:
        os.makedirs(os.path.dirname(ruta_gif), exist_ok=True)
        anim.save(ruta_gif, writer=PillowWriter(fps=max(1, 1000 // intervalo)))
        plt.close(fig)
        print(f"Animacion guardada en: {ruta_gif}")
        return ruta_gif

    plt.show()
    return None


def figura_estatica(datos, resumen, ruta=None, titulo=None):
    """
    Version fija del tablero: los mismos cuatro paneles con la corrida
    completa dibujada de una vez. Es la que va al informe, porque un GIF no
    entra en un PDF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = [nombre for nombre, *_ in CATALOGO_SECCIONES]
    segundos, fronteras = _eje_tiempo(datos)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.suptitle(titulo or "Boleteria bajo ataque - corrida completa", fontsize=14)
    ax_inv, ax_pre = axes[0]
    ax_acu, ax_col = axes[1]

    for k, nombre in enumerate(nombres):
        ax_inv.plot(segundos, datos["inventarios"][:, k], color=COLORES[k], lw=1.8, label=nombre)
        ax_pre.plot(segundos, datos["precios"][:, k], color=COLORES[k], lw=1.8, label=nombre)

    ax_acu.plot(segundos, datos["bots"], color="#c44e52", lw=2.2, label="Bots")
    ax_acu.plot(segundos, datos["humanos"], color="#4c72b0", lw=2.2, label="Humanos")
    ax_col.plot(segundos, datos["cola"], color="#8172b3", lw=1.8, label="Sala de espera")
    ax_col.plot(segundos, datos["cajas"], color="#55a868", lw=1.8, label="Cajas ocupadas (max 500)")

    titulos = ["Inventario por seccion", "Motor de precios (topes 0.7x - 3.0x)",
               "Quien se lleva el evento", "Cuello de botella: 500 cajas"]
    etiquetas_y = ["Boletos disponibles", "Precio vigente (Q)",
                   "Boletos comprados (acumulado)", "Entidades"]
    for ax, tit, ey in zip((ax_inv, ax_pre, ax_acu, ax_col), titulos, etiquetas_y):
        for frontera in fronteras:
            ax.axvline(frontera, color="gray", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(tit, fontsize=10)
        ax.set_ylabel(ey)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in (ax_acu, ax_col):
        ax.set_xlabel("Segundos de venta activa (las 3 fases empalmadas)")

    pie = (f"{resumen['boletos_vendidos']:,} boletos vendidos | "
           f"{resumen['fraccion_bots']:.1%} a bots | "
           f"Q {resumen['ingreso']:,.0f} de ingreso | "
           f"{resumen['abandonos_con_inventario']:,} abandonos con inventario disponible")
    fig.text(0.5, 0.005, pie, ha="center", fontsize=9, color="#444444")

    ruta = ruta or os.path.join(DIR_SALIDA, "tablero.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"Figura guardada en: {ruta}")
    return ruta


def imprimir_resumen(resumen):
    print("\n=== Resultado de la corrida del tablero ===")
    print(f"  Boletos vendidos          : {resumen['boletos_vendidos']:,}")
    print(f"    ...a bots               : {resumen['boletos_bots']:,} "
          f"({resumen['fraccion_bots']:.1%})")
    print(f"    ...a humanos            : {resumen['boletos_humanos']:,}")
    print(f"  Ingreso                   : Q {resumen['ingreso']:,.0f}")
    print(f"  Precio promedio           : Q {resumen['precio_promedio']:,.0f}")
    print(f"  Abandonos con inventario  : {resumen['abandonos_con_inventario']:,}")
    print(f"  Abandonos totales         : {resumen['abandonos_impaciencia']:,} "
          f"(la mayoria hacia fila cuando ya no quedaba nada)")
    print(f"  Se quedaron sin turno     : {resumen['sold_out_antes_del_turno']:,}")
    print(f"  Llegaron con todo agotado : {resumen['llegadas_con_agotado']:,}")
    print(f"  Bloqueados por politica   : {resumen['bloqueados_por_politica']:,}")


def main():
    parser = argparse.ArgumentParser(description="Tablero en vivo de la simulacion")
    parser.add_argument("--gif", action="store_true",
                        help="guarda resultados/tablero.gif en vez de abrir ventana")
    parser.add_argument("--eta", type=float, default=0.10,
                        help="ganancia del controlador de precios (0 = precio fijo)")
    parser.add_argument("--politica", default="ninguna",
                        help="politica anti-bot a aplicar (ver modelo/politicas.py)")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()

    politica = crear_politica(args.politica)
    print(f"Corriendo simulacion (eta={args.eta}, politica={args.politica})...")
    datos, resumen = correr_y_grabar(args.semilla, args.eta, politica)
    print(f"  {datos['t'].size:,} fotos grabadas")
    imprimir_resumen(resumen)

    figura_estatica(datos, resumen,
                    titulo=f"Boleteria bajo ataque - politica: {args.politica}, eta={args.eta}")

    ruta = os.path.join(DIR_SALIDA, "tablero.gif") if args.gif else None
    animar(datos, resumen, ruta_gif=ruta)


if __name__ == "__main__":
    main()
