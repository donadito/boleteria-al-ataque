from dataclasses import dataclass
from typing import List

DURACION_FASE = 1440.0  # Cada fase corre hasta la apertura de la siguiente

@dataclass
class FaseVenta:
    dia: int
    hora_apertura: float  # En minutos desde el día 1 a las 00:00 (día 2, 10:00 = 2040)
    liberacion_inventario: int
    poblacion_pool: int
    bots_admitidos: int   # Cuántos bots de la flota logran pasar el filtro de la fase
    nombre: str

def obtener_fases() -> List[FaseVenta]:
    return [
        FaseVenta(dia=1, hora_apertura=600, liberacion_inventario=10000, poblacion_pool=60000, bots_admitidos=3000, nombre="Preventa Fan Verificado"),
        FaseVenta(dia=2, hora_apertura=2040, liberacion_inventario=5000, poblacion_pool=45000, bots_admitidos=2000, nombre="Preventa Banco Aliado"),
        FaseVenta(dia=3, hora_apertura=3480, liberacion_inventario=10000, poblacion_pool=180000, bots_admitidos=22000, nombre="Venta General")
    ]
