class ControladorPrecios:
    def __init__(self, eta: float = 0.10, t_obj: float = 180.0):
        self.eta = eta
        self.t_obj = t_obj

    def actualizar_precios(self, secciones: list, liberacion_total: int):
        for sec in secciones:
            if sec.inventario > 0:
                # Ritmo necesario para agotar la sección
                objetivo = liberacion_total / self.t_obj 
                observado = sec.ventas_ultimo_minuto
                
                # Fórmula del controlador proporcional
                ajuste = 1 + self.eta * ((observado - objetivo) / objetivo)
                nuevo_precio = sec.precio_vigente * ajuste
                
                # Aplicar topes [0.7*po, 3.0*po]
                limite_inf = 0.7 * sec.precio_base
                limite_sup = 3.0 * sec.precio_base
                
                sec.precio_vigente = max(limite_inf, min(nuevo_precio, limite_sup))
                
                # Reiniciar el contador para la próxima ventana
                sec.ventas_ultimo_minuto = 0