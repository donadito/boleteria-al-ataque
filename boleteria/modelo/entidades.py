from dataclasses import dataclass
from typing import List

@dataclass
class Usuario:
    valuacion: float
    paciencia: float
    t_checkout: float
    fase: int
    t_llegada: float = 0.0
    cantidad: int = 0

@dataclass
class Bot:
    id: int = 0
    t_llegada: float = 0.0
    t_checkout: float = 0.05 # 3 segundos expresados en minutos
    paciencia: float = float('inf') # Sin límite de paciencia
    fase: int = 0
    cantidad: int = 4 # Compran el máximo permitido

@dataclass
class Seccion:
    nombre: str
    inventario: int
    precio_base: float
    calidad: float
    precio_vigente: float = 0.0
    ventas_ultimo_minuto: int = 0
    pagos_fallidos_retenidos: int = 0 # nueva propiedad para mapear los boletos retenidos por fallos que regresan al inventario
    liberado_fase: int = 0 # boletos que la fase en curso liberó a esta sección; es la base del ritmo objetivo

    def __post_init__(self):
        self.precio_vigente = self.precio_base
        if self.liberado_fase == 0:
            self.liberado_fase = self.inventario

# (nombre, capacidad, precio base, calidad q_k)
CATALOGO_SECCIONES = [
    ("General",      12000,  350.0, 1.0),
    ("Preferencial",  8000,  650.0, 1.6),
    ("VIP",           4000, 1200.0, 2.6),
    ("Palco",         1000, 2500.0, 4.5),
]
AFORO_TOTAL = 25000

def crear_secciones(liberacion: int) -> List[Seccion]:
    """Reparte el inventario que libera la fase proporcional a la capacidad de cada sección."""
    return [Seccion(nombre, round(liberacion * capacidad / AFORO_TOTAL), precio, calidad)
            for nombre, capacidad, precio, calidad in CATALOGO_SECCIONES]
