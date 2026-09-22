"""
Motor de la simulación: arreglos de tiempos procesados secuencialmente.

El diseño sigue lo que plantea la consigna del curso: se generan los arreglos
de tiempos (de llegada, de checkout) y se procesan en orden, en dos pasadas.
NO hay calendario de eventos, ni lista de eventos futuros, ni un reloj que
salte entre tipos de evento. Todo es recorrer arreglos ordenados.

PASADA 1 — Asignación de cajas.
    Se ordenan las llegadas por tiempo y se recorren una por una. Las 500 cajas
    de checkout se representan con un arreglo `libre_en[k]` que guarda a qué
    hora se desocupa cada caja. Para cada persona se busca la caja que se
    desocupa primero, y su checkout empieza en

        inicio = max(su tiempo de llegada, la hora en que esa caja se desocupa)

    Esa resta, `inicio - llegada`, es su espera, y es lo que se compara contra
    su paciencia: si esperó más de lo que aguantaba, se va sin comprar. Quien sí
    espera ocupa la caja hasta `inicio + su tiempo de checkout`.

PASADA 2 — Compra.
    La pasada anterior deja un arreglo de tiempos de finalización. Se ordena y
    se recorre: cada persona compra cuando TERMINA su checkout, no cuando
    empieza. Esa distinción es la que le da sentido al ataque, porque un bot
    paga en 3 s y un humano en 90 s: para cuando el humano acaba de teclear su
    tarjeta, el inventario puede haberse agotado.

El precio dinámico se recalcula avanzando por ventanas de `paso_precio`
minutos conforme se recorre el arreglo de finalizaciones, lo que le da al
controlador el ritmo de venta de cada ventana sin necesidad de un reloj global.

Sobre por qué basta un arreglo de 500 marcas: la alternativa de asumir que la
caja libre es siempre la del recorrido (`fin[i - 500]`) daría resultados falsos,
porque los tiempos de checkout son muy desiguales (3 s contra 90 s) y las cajas
NO se desocupan en el mismo orden en que se ocuparon. Buscar el mínimo del
arreglo en cada paso es exacto y cuesta una comparación vectorizada sobre 500
números.
"""
import numpy as np

from .calendario import DURACION_FASE
from .entidades import Bot

UN_SEGUNDO = 1.0 / 60.0   # el reloj del modelo corre en minutos

# Tope de ventanas de precio que se recalculan de golpe entre dos compras. Sin
# él, un hueco largo sin ventas haría girar el controlador cientos de miles de
# veces sin cambiar nada: tras unas decenas de ventanas vacías el precio ya
# tocó el piso y ahí se queda.
MAX_VENTANAS_SEGUIDAS = 500


class MotorSimulacion:
    def __init__(self, generador_normal, generador_uniforme,
                 slots_concurrentes: int = 500,
                 paso_precio: float = UN_SEGUNDO,
                 prob_fallo_pago: float = 0.07,
                 politica=None):
        self.gen_normal = generador_normal
        self.gen_uniform = generador_uniforme
        self.slots_concurrentes = slots_concurrentes
        self.paso_precio = paso_precio
        self.prob_fallo_pago = prob_fallo_pago
        # Política anti-bot opcional (ver modelo/politicas.py). None = sin defensa.
        self.politica = politica

        self.inventario_global = 0
        self.t_agotado = None     # instante en que se agotó la fase, si se agotó

        # Arreglos de la última fase corrida, que el tablero usa para dibujar.
        self.llegadas_ordenadas = np.empty(0)
        self.inicios_servidos = np.empty(0)
        self.fines_servidos = np.empty(0)
        self.tiempos_abandono = np.empty(0)

        self.estadisticas = {
            'ingreso': 0.0,
            'boletos_vendidos': 0,
            'boletos_bots': 0,
            'boletos_humanos': 0,
            'excedente_consumidor': 0.0,
            'abandonos_impaciencia': 0,
            'abandonos_con_inventario': 0,
            'salidas_por_precio': 0,
            'sold_out_antes_del_turno': 0,
            'llegadas_con_agotado': 0,
            'pagos_rechazados': 0,
            'bloqueados_por_politica': 0,
            'bloqueados_bots': 0,
            'bloqueados_humanos': 0,
            'atendidos': 0,
            'espera_total': 0.0,
            'espera_maxima': 0.0,
        }

    # ------------------------------------------------------------------ pagos

    def simular_pago(self) -> bool:
        """El pago falla con probabilidad `prob_fallo_pago`."""
        u = self.gen_uniform.uniform(1)[0]
        return u > self.prob_fallo_pago

    # ------------------------------------------------------------------ fases

    def correr_fase(self, fase, secciones, controlador, llegadas, observador=None):
        """
        Corre una fase completa: una pasada para asignar cajas y otra para comprar.

        `llegadas` es la lista de entidades ya generadas (humanos del NHPP más la
        ráfaga de bots). `observador`, si se pasa, se invoca en cada compra con
        (tiempo, motor, secciones) y es lo que alimenta al tablero; el
        observador decide cada cuánto guardar.
        """
        if self.politica is not None:
            llegadas, bloqueos = self.politica.filtrar(llegadas, fase, self.gen_uniform)
            self.estadisticas['bloqueados_bots'] += bloqueos["bots"]
            self.estadisticas['bloqueados_humanos'] += bloqueos["humanos"]
            self.estadisticas['bloqueados_por_politica'] += bloqueos["bots"] + bloqueos["humanos"]

        self.inventario_global = sum(sec.inventario for sec in secciones)
        orden_llegada = sorted(llegadas, key=lambda entidad: entidad.t_llegada)

        servidos = self._asignar_cajas(orden_llegada, fase)
        self._procesar_compras(servidos, secciones, controlador, fase, observador)
        return dict(self.estadisticas)

    def _asignar_cajas(self, orden_llegada, fase):
        """
        Pasada 1: recorre las llegadas en orden y le da a cada una la primera
        caja que se desocupe, descartando a quien pierde la paciencia esperando.

        Devuelve la lista de (tiempo_de_finalizacion, entidad) de quienes sí
        alcanzaron a hacer checkout.
        """
        cierre = fase.hora_apertura + DURACION_FASE
        # Todas las cajas arrancan libres en el momento de la apertura.
        libre_en = np.full(self.slots_concurrentes, fase.hora_apertura, dtype=np.float64)

        servidos = []
        inicios, fines, abandonos = [], [], []

        for entidad in orden_llegada:
            caja = int(np.argmin(libre_en))
            inicio = max(entidad.t_llegada, float(libre_en[caja]))
            if inicio > cierre:
                break   # la ventana de venta cerró

            espera = inicio - entidad.t_llegada
            if espera > entidad.paciencia:
                self.estadisticas['abandonos_impaciencia'] += 1
                abandonos.append(inicio)
                continue

            fin = inicio + entidad.t_checkout
            libre_en[caja] = fin

            self.estadisticas['atendidos'] += 1
            self.estadisticas['espera_total'] += espera
            if espera > self.estadisticas['espera_maxima']:
                self.estadisticas['espera_maxima'] = espera

            servidos.append((fin, entidad))
            inicios.append(inicio)
            fines.append(fin)

        self.llegadas_ordenadas = np.array([e.t_llegada for e in orden_llegada])
        self.inicios_servidos = np.sort(np.array(inicios))
        self.fines_servidos = np.sort(np.array(fines))
        self.tiempos_abandono = np.sort(np.array(abandonos))
        return servidos

    def _procesar_compras(self, servidos, secciones, controlador, fase, observador):
        """
        Pasada 2: recorre las finalizaciones en orden y concreta cada compra,
        avanzando el precio por ventanas conforme corre el arreglo.
        """
        servidos.sort(key=lambda par: par[0])
        proximo_tick = fase.hora_apertura + self.paso_precio

        for i, (tiempo, entidad) in enumerate(servidos):
            if self.inventario_global <= 0:
                self.estadisticas['sold_out_antes_del_turno'] += len(servidos) - i
                self._cerrar_agotado(tiempo)
                return

            # Se cierran todas las ventanas de precio que terminaron antes de
            # esta compra, para que el controlador vea el ritmo de cada una.
            ventanas = 0
            while proximo_tick <= tiempo and ventanas < MAX_VENTANAS_SEGUIDAS:
                controlador.actualizar_precios(secciones, self.paso_precio)
                proximo_tick += self.paso_precio
                ventanas += 1
            if ventanas >= MAX_VENTANAS_SEGUIDAS:
                proximo_tick = tiempo + self.paso_precio

            self.ejecutar_decision_compra(entidad, secciones, tiempo)

            if observador is not None:
                observador(tiempo, self, secciones)

        self._cerrar_agotado(None)

    def _cerrar_agotado(self, tiempo):
        """
        Separa los abandonos que ocurrieron mientras todavía había boletos de los
        que ocurrieron después.

        La pasada 1 forma la fila con TODAS las llegadas de la fase, porque no
        puede saber de antemano cuándo se agota el inventario -- eso lo decide
        la pasada 2. El resultado es que la mayoría de los abandonos son de
        gente que hacía fila cuando ya no quedaba nada, y contarlos junto con
        los demás daría una cifra engañosa. La asignación de boletos no cambia
        (se la llevan las primeras finalizaciones), pero la estadística sí, así
        que se reportan por separado.
        """
        self.t_agotado = tiempo
        if tiempo is None:
            # La fase cerró con inventario: todos los abandonos son "con boletos".
            self.estadisticas['abandonos_con_inventario'] += int(self.tiempos_abandono.size)
            return
        antes = int(np.searchsorted(self.tiempos_abandono, tiempo, side="right"))
        self.estadisticas['abandonos_con_inventario'] += antes

    def correr_simulacion_completa(self, fases, secciones, controlador,
                                   generador_llegadas, observador=None):
        """
        Corre los tres días de venta y la ventana de devoluciones.

        `generador_llegadas` es un callable fase -> lista de entidades; así el
        motor no depende de cómo se construyen las llegadas (NHPP, flota de
        bots, política de acceso) y los experimentos pueden inyectar lo suyo.
        """
        for fase in fases:
            liberar_inventario(secciones, fase.liberacion_inventario)
            llegadas = generador_llegadas(fase)
            self.correr_fase(fase, secciones, controlador, llegadas, observador)

        # Días 4 y 5: los boletos retenidos por pagos fallidos vuelven al inventario.
        reinyectados = 0
        for sec in secciones:
            if sec.pagos_fallidos_retenidos > 0:
                sec.inventario += sec.pagos_fallidos_retenidos
                reinyectados += sec.pagos_fallidos_retenidos
                sec.pagos_fallidos_retenidos = 0
        self.estadisticas['boletos_reinyectados'] = reinyectados

        return dict(self.estadisticas)

    # ------------------------------------------------------------------ compra

    def ejecutar_decision_compra(self, entidad, secciones, tiempo_salida):
        """
        El usuario calcula su excedente V*q_k - p_k en cada sección con inventario y se queda
        con la que se lo maximiza. Si ni la mejor le deja excedente positivo, se va sin comprar.
        """
        disponibles = [sec for sec in secciones if sec.inventario > 0]
        if not disponibles:
            self.estadisticas['sold_out_antes_del_turno'] += 1
            return

        if isinstance(entidad, Bot):
            # Los bots ignoran el precio: se llevan la mejor localidad que quede
            elegida = max(disponibles, key=lambda sec: sec.calidad)
            excedente = 0.0
        else:
            excedente_de = lambda sec: entidad.valuacion * sec.calidad - sec.precio_vigente
            elegida = max(disponibles, key=excedente_de)
            excedente = excedente_de(elegida)
            if excedente <= 0:
                self.estadisticas['salidas_por_precio'] += 1
                return

        tope = entidad.cantidad
        if self.politica is not None:
            tope = min(tope, self.politica.tope_compra(entidad))
        cantidad = min(tope, elegida.inventario)
        if cantidad <= 0:
            self.estadisticas['sold_out_antes_del_turno'] += 1
            return

        elegida.inventario -= cantidad
        self.inventario_global -= cantidad

        if not self.simular_pago():
            # Los boletos quedan retenidos hasta la ventana de devoluciones de los días 4-5
            elegida.pagos_fallidos_retenidos += cantidad
            self.estadisticas['pagos_rechazados'] += 1
            return

        elegida.ventas_ultimo_minuto += cantidad
        self.estadisticas['ingreso'] += cantidad * elegida.precio_vigente
        self.estadisticas['boletos_vendidos'] += cantidad
        if isinstance(entidad, Bot):
            self.estadisticas['boletos_bots'] += cantidad
        else:
            self.estadisticas['boletos_humanos'] += cantidad
            self.estadisticas['excedente_consumidor'] += excedente

    # ------------------------------------------------------- estado para el tablero

    def cajas_ocupadas_en(self, t: float) -> int:
        """Cuántas cajas están en uso en el instante t: empezaron antes y no han terminado."""
        return int(np.searchsorted(self.inicios_servidos, t, side="right")
                   - np.searchsorted(self.fines_servidos, t, side="right"))

    def en_espera_en(self, t: float) -> int:
        """
        Cuánta gente hay en la sala de espera en el instante t: llegó, todavía no
        entra a una caja y todavía no se ha ido por impaciencia.
        """
        llegaron = np.searchsorted(self.llegadas_ordenadas, t, side="right")
        entraron = np.searchsorted(self.inicios_servidos, t, side="right")
        se_fueron = np.searchsorted(self.tiempos_abandono, t, side="right")
        return int(max(0, llegaron - entraron - se_fueron))

    # ----------------------------------------------------------------- reporte

    def resumen(self) -> dict:
        """Estadísticas más las métricas derivadas que usan los experimentos."""
        est = dict(self.estadisticas)
        atendidos = est['atendidos']
        vendidos = est['boletos_vendidos']
        est['espera_promedio'] = est['espera_total'] / atendidos if atendidos else 0.0
        est['fraccion_bots'] = est['boletos_bots'] / vendidos if vendidos else 0.0
        est['precio_promedio'] = est['ingreso'] / vendidos if vendidos else 0.0
        return est


def liberar_inventario(secciones, liberacion: int):
    """
    Agrega a las secciones persistentes el inventario que libera una fase y
    reinicia su cuota objetivo, que es la base del ritmo que persigue el
    controlador de precios.
    """
    from .entidades import crear_secciones

    for sec, delta in zip(secciones, crear_secciones(liberacion)):
        sec.inventario += delta.inventario
        sec.liberado_fase = delta.inventario
