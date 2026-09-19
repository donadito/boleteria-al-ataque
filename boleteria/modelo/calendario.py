from dataclasses import dataclass
from typing import List

@dataclass
class FaseVenta:
    dia: int
    hora_apertura: float  # En minutos desde la medianoche (10:00 = 600)
    liberacion_inventario: int
    poblacion_pool: int
    nombre: str

def obtener_fases() -> List[FaseVenta]:
    return [
        FaseVenta(dia=1, hora_apertura=600, liberacion_inventario=10000, poblacion_pool=60000, nombre="Preventa Fan Verificado"),
        FaseVenta(dia=2, hora_apertura=600, liberacion_inventario=5000, poblacion_pool=45000, nombre="Preventa Banco Aliado"),
        FaseVenta(dia=3, hora_apertura=600, liberacion_inventario=10000, poblacion_pool=100000, nombre="Venta General")
    ]