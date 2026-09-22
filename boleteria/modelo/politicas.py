"""
Politicas de defensa contra la automatizacion de compra.

Sin defensa, el resultado del modelo es de una sola linea: los bots se llevan
practicamente el 100% del inventario. No es un desbalance de parametros, es
mecanico: un bot paga en 3 s y un humano en 90 s, y como la compra se concreta
al TERMINAR el checkout, para cuando el primer humano acaba de teclear su
tarjeta ya no queda nada. Este modulo modela las cuatro defensas que usa la
industria y deja que el experimento 3 las compare.

Cada politica actua en uno de dos momentos:

    filtrar()      antes de que empiece la fase: decide quien entra y con que
                   tiempo de llegada (bloqueos, sorteo de turno)
    tope_compra()  en el momento de comprar: cuantos boletos puede llevarse

Las cuatro politicas modeladas:

  CupoEstricto      Bajar el tope de boletos por cuenta de 4 a 2. Es la medida
                    mas barata y no molesta a casi nadie, pero un atacante con
                    22,000 cuentas simplemente necesita el doble de cuentas.

  Captcha           Un filtro de deteccion que bloquea a la mayoria de los bots
                    a cambio de bloquear tambien a algunos humanos legitimos
                    (falsos positivos). El intercambio entre tasa de deteccion
                    y falso positivo es TODO el problema: un captcha que
                    bloquea 99% de bots y 10% de humanos puede ser peor que
                    uno que bloquea 90% y 1%.

  LoteriaDeTurno    La sala de espera con turno sorteado. En vez de atender por
                    orden de llegada, todos los que llegan en la ventana de
                    apertura reciben una posicion aleatoria dentro de una
                    ventana mas larga. Destruye la ventaja de conectarse en el
                    segundo cero, que es la ventaja que el modelo le da a los
                    bots. Es lo que hacen los sistemas de boleteria reales.

  Combinada         Loteria + cupo estricto, para ver si las defensas se suman
                    o se estorban.

Ninguna de estas politicas toca el motor: el motor solo pregunta.
"""


class Politica:
    """Politica base: deja pasar a todos y no cambia el tope de compra."""

    nombre = "ninguna"
    descripcion = "Sin defensa"

    def filtrar(self, llegadas, fase, gen_uniforme):
        """
        Devuelve (llegadas_admitidas, bloqueos), con bloqueos desglosado en
        {"bots": n, "humanos": m}. El desglose no es cosmetico: bloquear bots
        es el beneficio de la defensa y bloquear humanos es su costo, y
        sumarlos en un solo numero esconde exactamente el intercambio que el
        experimento 3 quiere medir.
        """
        return llegadas, {"bots": 0, "humanos": 0}

    def tope_compra(self, entidad) -> int:
        """Maximo de boletos que esta entidad puede llevarse en una compra."""
        return entidad.cantidad


def _es_bot(entidad) -> bool:
    # Se identifica por el atributo, no por isinstance, para no importar
    # entidades aqui y evitar el ciclo de importacion con el motor.
    return hasattr(entidad, "id") and not hasattr(entidad, "valuacion")


class CupoEstricto(Politica):
    """Baja el tope de boletos por cuenta. Afecta a humanos y bots por igual."""

    nombre = "cupo_estricto"

    def __init__(self, tope: int = 2):
        self.tope = int(tope)
        self.descripcion = f"Tope de {self.tope} boletos por cuenta (base: 4)"

    def tope_compra(self, entidad) -> int:
        return min(entidad.cantidad, self.tope)


class Captcha(Politica):
    """
    Filtro de deteccion con falsos positivos.

    `deteccion` es la fraccion de bots que bloquea; `falso_positivo` la
    fraccion de humanos legitimos que bloquea por error. Los dos numeros van
    juntos: subir la deteccion sin subir el falso positivo no es gratis en la
    vida real, y el experimento 3 barre el intercambio.
    """

    nombre = "captcha"

    def __init__(self, deteccion: float = 0.95, falso_positivo: float = 0.03):
        self.deteccion = float(deteccion)
        self.falso_positivo = float(falso_positivo)
        self.descripcion = (f"Captcha: bloquea {self.deteccion:.0%} de bots "
                            f"y {self.falso_positivo:.0%} de humanos")

    def filtrar(self, llegadas, fase, gen_uniforme):
        if not llegadas:
            return llegadas, {"bots": 0, "humanos": 0}
        u = gen_uniforme.uniform(len(llegadas))
        admitidas = []
        bloqueos = {"bots": 0, "humanos": 0}
        for entidad, sorteo in zip(llegadas, u):
            es_bot = _es_bot(entidad)
            umbral = self.deteccion if es_bot else self.falso_positivo
            if sorteo < umbral:
                bloqueos["bots" if es_bot else "humanos"] += 1
            else:
                admitidas.append(entidad)
        return admitidas, bloqueos


class LoteriaDeTurno(Politica):
    """
    Sala de espera con turno sorteado.

    Todo el que llega dentro de `ventana_entrada` minutos de la apertura recibe
    un turno uniforme dentro de `ventana_sorteo` minutos. Conectarse en el
    segundo cero deja de servir de nada: el bot que llego en el segundo 0.3 y
    el humano que llego en el minuto 4 tienen exactamente la misma
    probabilidad de quedar al frente.

    Los que llegan despues de la ventana de entrada conservan su tiempo real,
    porque ya no compiten por el sorteo.
    """

    nombre = "loteria"

    def __init__(self, ventana_entrada: float = 5.0, ventana_sorteo: float = 30.0):
        self.ventana_entrada = float(ventana_entrada)
        self.ventana_sorteo = float(ventana_sorteo)
        self.descripcion = (f"Loteria: turno sorteado en {self.ventana_sorteo:.0f} min "
                            f"para quien entra en los primeros {self.ventana_entrada:.0f} min")

    def filtrar(self, llegadas, fase, gen_uniforme):
        if not llegadas:
            return llegadas, {"bots": 0, "humanos": 0}

        limite = fase.hora_apertura + self.ventana_entrada
        en_sorteo = [e for e in llegadas if e.t_llegada <= limite]
        if en_sorteo:
            turnos = gen_uniforme.uniform(len(en_sorteo)) * self.ventana_sorteo
            for entidad, turno in zip(en_sorteo, turnos):
                entidad.t_llegada = fase.hora_apertura + float(turno)

        return llegadas, {"bots": 0, "humanos": 0}


class Combinada(Politica):
    """Aplica varias politicas en cadena: filtros encadenados, tope mas estricto."""

    nombre = "combinada"

    def __init__(self, politicas):
        self.politicas = list(politicas)
        self.descripcion = " + ".join(p.descripcion for p in self.politicas)

    def filtrar(self, llegadas, fase, gen_uniforme):
        total = {"bots": 0, "humanos": 0}
        for politica in self.politicas:
            llegadas, bloqueos = politica.filtrar(llegadas, fase, gen_uniforme)
            total["bots"] += bloqueos["bots"]
            total["humanos"] += bloqueos["humanos"]
        return llegadas, total

    def tope_compra(self, entidad) -> int:
        return min(p.tope_compra(entidad) for p in self.politicas)


CATALOGO = {
    "ninguna": lambda: Politica(),
    "cupo_estricto": lambda: CupoEstricto(tope=2),
    "captcha": lambda: Captcha(deteccion=0.95, falso_positivo=0.03),
    "loteria": lambda: LoteriaDeTurno(),
    "combinada": lambda: Combinada([LoteriaDeTurno(), CupoEstricto(tope=2)]),
}


def crear_politica(nombre: str):
    """
    Fabrica una politica por nombre. Devuelve None para 'ninguna', que es lo
    que el motor entiende como "no preguntes nada" y evita el costo de llamar
    a un filtro que no hace nada en cada fase.
    """
    if nombre in (None, "", "ninguna"):
        return None
    if nombre not in CATALOGO:
        raise ValueError(f"Politica desconocida: {nombre!r}. "
                         f"Opciones: {', '.join(CATALOGO)}")
    return CATALOGO[nombre]()
