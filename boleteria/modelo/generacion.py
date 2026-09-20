import numpy as np
from typing import List
from .calendario import DURACION_FASE, FaseVenta
from .entidades import Bot, Usuario

SEGUNDO = 1 / 60.0        # El reloj del modelo corre en minutos
RAFAGA_BOTS = 5 * SEGUNDO # Los bots se conectan en el segundo cero: [T, T + 5 s]

# Boletos por compra: discreta empírica 1..4 con p = .18 .52 .10 .20
CDF_BOLETOS = np.cumsum([0.18, 0.52, 0.10, 0.20])

def _boletos_por_compra(gen_uniforme, n: int) -> np.ndarray:
    # Búsqueda en la CDF
    return np.searchsorted(CDF_BOLETOS, gen_uniforme.uniform(n)) + 1

def generar_usuarios(fase: FaseVenta, nhpp, gen_normal, gen_uniforme,
                     duracion: float = DURACION_FASE) -> List[Usuario]:
    """Cada llegada del proceso de Poisson no homogéneo se vuelve un Usuario."""
    tiempos = nhpp.generar_fase(fase.hora_apertura, duracion, fase.poblacion_pool)
    n = len(tiempos)
    if n == 0:
        return []

    valuacion = gen_normal.normal_truncada(n, 500.0, 220.0, 50.0)
    paciencia = gen_normal.normal_truncada(n, 20.0, 8.0, 1.0)
    checkout = gen_normal.normal_truncada(n, 90.0, 30.0, 15.0, 600.0) * SEGUNDO
    cantidad = _boletos_por_compra(gen_uniforme, n)

    return [Usuario(valuacion=float(v), paciencia=float(p), t_checkout=float(c),
                    fase=fase.dia, t_llegada=float(t), cantidad=int(q))
            for t, v, p, c, q in zip(tiempos, valuacion, paciencia, checkout, cantidad)]

def crear_flota_bots(n_bots: int, gen_normal) -> List[Bot]:
    """
    La botnet se crea UNA sola vez para toda la simulación: los bots no se reinician
    entre fases, conservan su identidad y su tiempo de checkout. Lo que cambia por fase
    es cuántos de ellos pasan el filtro de acceso.
    """
    checkout = gen_normal.normal_truncada(n_bots, 3.0, 1.0, 1.0) * SEGUNDO
    return [Bot(id=i, t_checkout=float(c)) for i, c in enumerate(checkout)]

def activar_bots(flota: List[Bot], fase: FaseVenta, gen_uniforme) -> List[Bot]:
    """Los bots admitidos entran en ráfaga en [T, T + 5 s]; el resto no juega esta fase."""
    admitidos = flota[:fase.bots_admitidos]
    desfases = gen_uniforme.uniform(len(admitidos)) * RAFAGA_BOTS

    for bot, desfase in zip(admitidos, desfases):
        bot.fase = fase.dia
        bot.t_llegada = fase.hora_apertura + float(desfase)
        bot.cantidad = 4 # El tope de 4 boletos es por cuenta y por fase
    return admitidos

def generar_llegadas(fase: FaseVenta, flota: List[Bot], nhpp, gen_normal, gen_uniforme) -> list:
    """Fila de entrada de la fase: humanos del NHPP más la ráfaga de bots, en orden de llegada."""
    llegadas = generar_usuarios(fase, nhpp, gen_normal, gen_uniforme)
    llegadas.extend(activar_bots(flota, fase, gen_uniforme))
    llegadas.sort(key=lambda entidad: entidad.t_llegada)
    return llegadas
