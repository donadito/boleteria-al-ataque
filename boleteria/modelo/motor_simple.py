"""
Variantes reducidas del modelo de atencion, para medir cuanto detalle hace falta.

El motor de motor.py procesa arreglos de tiempos en dos pasadas y representa el
recurso limitado: las 500 cajas de checkout, con su arreglo de horas de
liberacion, la espera que eso genera y el abandono por impaciencia.

Este modulo implementa dos versiones mas simples del MISMO recorrido
secuencial, que se van quitando piezas del modelo. Sirven para el experimento 4:
responder con numeros, y no con opiniones, si representar el recurso valia la
pena o si bastaba con ordenar y recorrer.

    MotorSecuencialLlegada
        Ordena por tiempo de llegada y recorre. No hay cajas, no hay espera y
        por lo tanto no hay abandono por impaciencia: los tiempos de checkout
        se generan pero no afectan nada. Quien llega primero, compra primero.

    MotorSecuencialFinalizacion
        Ordena por llegada + tiempo de checkout y recorre. Un cambio de una
        linea frente al anterior, pero ahora el tiempo de checkout SI decide, y
        con eso captura que el bot paga en 3 s y el humano en 90 s. Sigue sin
        haber recurso limitado ni espera.

    MotorSimulacion (motor.py)
        Ademas de lo anterior, representa las 500 cajas: quien llega cuando
        estan todas ocupadas espera, y si espera mas que su paciencia se va.

Lo que las dos variantes reducidas NO pueden producir, y conviene tenerlo claro
antes de leer la comparacion:

  * Abandono por impaciencia. Sin cajas no hay espera, asi que la paciencia
    -- una de las tres variables aleatorias del modelo, con su normal truncada
    y su validacion -- se genera y no se usa.
  * Longitud de la sala de espera y tiempo de espera, que son dos de los cuatro
    paneles del tablero.
  * Ritmo de venta por ventana. Sin saber cuando termina cada checkout no hay
    contra que medir el ritmo, asi que el precio dinamico no es representable.
    Por eso la comparacion del experimento 4 se corre con precio FIJO en los
    tres: comparar con precio dinamico seria comparar modelos distintos y
    atribuirselo al orden de atencion.
"""
from .motor import MotorSimulacion


class MotorSecuencial(MotorSimulacion):
    """
    Base de las variantes sin recurso: ordenar el arreglo y recorrerlo.

    Reemplaza las dos pasadas del motor completo por una sola, porque sin cajas
    no hay nada que asignar. Las subclases solo definen `clave_orden`, que es
    toda la diferencia entre ellas.
    """

    nombre = "secuencial"

    def clave_orden(self, entidad):
        raise NotImplementedError

    def correr_fase(self, fase, secciones, controlador, llegadas, observador=None):
        if self.politica is not None:
            llegadas, bloqueos = self.politica.filtrar(llegadas, fase, self.gen_uniform)
            self.estadisticas['bloqueados_bots'] += bloqueos["bots"]
            self.estadisticas['bloqueados_humanos'] += bloqueos["humanos"]
            self.estadisticas['bloqueados_por_politica'] += bloqueos["bots"] + bloqueos["humanos"]

        self.inventario_global = sum(sec.inventario for sec in secciones)
        orden = sorted(llegadas, key=self.clave_orden)

        for i, entidad in enumerate(orden):
            if self.inventario_global <= 0:
                # Los que quedan llegaron a encontrar el letrero de agotado.
                self.estadisticas['llegadas_con_agotado'] += len(orden) - i
                break
            self.estadisticas['atendidos'] += 1
            self.ejecutar_decision_compra(entidad, secciones, self.clave_orden(entidad))

        return dict(self.estadisticas)


class MotorSecuencialLlegada(MotorSecuencial):
    """Quien llega primero compra primero. El tiempo de checkout no se usa."""

    nombre = "solo_llegada"

    def clave_orden(self, entidad):
        return entidad.t_llegada


class MotorSecuencialFinalizacion(MotorSecuencial):
    """
    Quien TERMINA de pagar primero compra primero.

    Un cambio de una linea frente al anterior, pero incorpora la asimetria que
    define el ataque: el bot cierra su compra 30 veces mas rapido que el
    humano. Sigue sin haber recurso limitado.
    """

    nombre = "llegada_y_checkout"

    def clave_orden(self, entidad):
        return entidad.t_llegada + entidad.t_checkout


MOTORES = {
    "solo_llegada": MotorSecuencialLlegada,
    "llegada_y_checkout": MotorSecuencialFinalizacion,
    "con_cajas": MotorSimulacion,
}
