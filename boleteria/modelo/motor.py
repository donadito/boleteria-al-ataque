import heapq
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

    def procesar_cola_espera(self, tiempo_actual):
        """
        Procesa a los usuarios en la cola bajo una política FIFO (porque si fuera LIFO, que mala onda) estricta,
        manejando el límite de 500 slots concurrentes en el checkout.
        """
        # Mientras haya gente en la fila y boletos disponibles
        while self.cola_espera and self.inventario_global > 0:
            usuario = self.cola_espera.pop(0)  # FIFO: sacamos al primero de la fila
            
            # 1. ENCONTRAR UN SLOT LIBRE
            if len(self.relojes_checkout) < self.slots_concurrentes:
                # Si hay menos de 500 personas pagando, el slot se asigna de inmediato
                tiempo_inicio_checkout = max(tiempo_actual, usuario.t_llegada)
            else:
                # Si los 500 slots están llenos, buscamos el que se va a desocupar más pronto.
                # heapq.heappop saca el tiempo más bajo de finalización de checkout con el metodo pop (hasta arribba)
                tiempo_slot_liberado = heapq.heappop(self.relojes_checkout)
                # El usuario inicia su checkout en cuanto se libera el slot (o en su tiempo de llegada, si el slot se liberó en el pasado relativo).
                tiempo_inicio_checkout = max(tiempo_slot_liberado, usuario.t_llegada)
            
            # 2. EVALUAR PACIENCIA. Esto aun falta de programar, pero sigue la distribución normal TRUNCADAAAA
            tiempo_espera = tiempo_inicio_checkout - usuario.t_llegada
            if tiempo_espera > usuario.paciencia:
                self.estadisticas['abandonos_impaciencia'] += 1
                continue # El usuario se va, pasamos al siguiente en la fila
            
            # 3. OCUPAR EL SLOT Y HACER CHECKOUT
            tiempo_salida = tiempo_inicio_checkout + usuario.t_checkout
            
            # Volvemos a meter este slot ocupado al heap con su nuevo tiempo de salida
            heapq.heappush(self.relojes_checkout, tiempo_salida)
            
            # 4. PASAR A LA DECISIÓN DE COMPRA
            # (Aquí falta llamar el método que evalúa el excedente y descuenta inventario)
            self.ejecutar_decision_compra(usuario, tiempo_salida)