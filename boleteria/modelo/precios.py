"""
Controlador proporcional de precio dinámico, sección por sección.

El controlador compara el RITMO de venta observado en la ventana de control
contra el ritmo objetivo y mueve el precio en esa dirección:

    ajuste = 1 + eta * (observado - objetivo) / objetivo

Ambos términos tienen que estar en las mismas unidades. El objetivo se define
como "agotar lo que la fase liberó a esta sección en t_obj minutos", así que el
ritmo objetivo es liberado_fase / t_obj boletos por minuto, y lo que le
corresponde a una ventana de dt minutos es (liberado_fase / t_obj) * dt. Contra
eso se compara `ventas_ultimo_minuto`, que es lo vendido DENTRO de esa ventana.

Dos detalles que importan:

  * El objetivo es POR SECCIÓN, no global. Palco libera 400 boletos y General
    4,800; medir ambas contra el mismo número haría que Palco pareciera siempre
    lenta y que su precio cayera al piso sin razón.
  * El objetivo escala con dt. Si no lo hiciera, cambiar la frecuencia de
    control cambiaría el resultado aunque el sistema fuera idéntico.

El precio queda acotado a [0.7*p0, 3.0*p0] para que el controlador no pueda
regalar ni extorsionar sin límite.
"""


class ControladorPrecios:
    def __init__(self, eta: float = 0.10, t_obj: float = 180.0,
                 piso: float = 0.7, techo: float = 3.0):
        self.eta = eta
        self.t_obj = t_obj
        self.piso = piso
        self.techo = techo

    def actualizar_precios(self, secciones: list, dt: float):
        """Recalcula el precio de cada sección tras una ventana de `dt` minutos."""
        for sec in secciones:
            # Una sección agotada no tiene precio que ajustar, pero su contador
            # sí se reinicia: si vuelve a tener inventario (devoluciones de los
            # días 4-5) debe arrancar con la ventana limpia.
            if sec.inventario <= 0 or sec.liberado_fase <= 0 or self.eta == 0.0:
                sec.ventas_ultimo_minuto = 0
                continue

            objetivo = sec.liberado_fase / self.t_obj * dt
            observado = sec.ventas_ultimo_minuto

            ajuste = 1 + self.eta * ((observado - objetivo) / objetivo)
            nuevo_precio = sec.precio_vigente * ajuste

            limite_inf = self.piso * sec.precio_base
            limite_sup = self.techo * sec.precio_base
            sec.precio_vigente = max(limite_inf, min(nuevo_precio, limite_sup))

            sec.ventas_ultimo_minuto = 0
