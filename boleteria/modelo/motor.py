class MotorSimulacion:
    def __init__(self, generador_normal, generador_uniforme):
        self.gen_normal = generador_normal
        self.gen_uniform = generador_uniforme
        
        self.slots_concurrentes = 500
        self.relojes_checkout = [] 
        self.cola_espera = []
        
    def simular_pago(self) -> bool:
        # El pago falla con probabilidad 0.07
        u = self.gen_uniform.uniform(1)[0]
        return u > 0.07

    def correr_fase(self, fase, secciones, controlador):
        print(f"== Iniciando {fase.nombre} ==")
        tiempo_actual = fase.hora_apertura
        # (Aquí irá la lógica de llegadas, checkout y abandono por paciencia)

    def correr_simulacion_completa(self, fases, secciones, controlador):
        # 1. Correr los 3 días de venta
        for fase in fases:
            self.correr_fase(fase, secciones, controlador)
            
        # 2. Día 4 y 5: Ventana de devoluciones
        print("== Iniciando Ventana de Devoluciones (Días 4-5) ==")
        for sec in secciones:
            if sec.pagos_fallidos_retenidos > 0:
                sec.inventario += sec.pagos_fallidos_retenidos
                print(f"Reinyectando {sec.pagos_fallidos_retenidos} boletos a {sec.nombre}")
                sec.pagos_fallidos_retenidos = 0
                
        # (Aquí VAMOS A METER el bucle final para vender las devoluciones con el tráfico de fondo)
