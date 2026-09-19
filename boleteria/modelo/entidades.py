from dataclasses import dataclass

@dataclass
class Usuario:
    valuacion: float
    paciencia: float
    t_checkout: float
    fase: int
    cantidad: int = 0

@dataclass
class Bot:
    t_checkout: float = 3.0 # El bot tarda 3 segundos fijos
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
    
    def __post_init__(self):
        self.precio_vigente = self.precio_base