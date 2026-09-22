"""
Generadores uniformes base: el LCG propio del grupo y el Mersenne Twister.

Todo lo demas del proyecto (normales por Polar, por Rechazo, por inversa, y el
proceso de Poisson) se construye sobre uno de estos dos. Esta es la "capa 1"
del modelo: si los uniformes estan mal, todo lo de arriba esta mal, y por eso
validacion.py los prueba por separado antes que nada.

Sobre los parametros. Un LCG es x_{n+1} = (a*x_n + c) mod m, y la calidad
depende por completo de la terna (a, c, m): no se pueden mezclar parametros de
generadores distintos. La version original de este archivo usaba el
multiplicador y el incremento de glibc (a = 1103515245, c = 12345), que estan
disenados para m = 2^31, junto con m = 2^31 - 1, que es primo. Esa combinacion
no tiene garantia de periodo completo ni esta estudiada en la literatura, asi
que se corrigio: el default ahora es MINSTD (Park-Miller con a = 48271), que si
es una terna estandar para m = 2^31 - 1, y las demas se ofrecen como presets
explicitos.

RANDU se incluye a proposito como control DEFECTUOSO. Es el generador de IBM de
los anos 60, famoso porque sus ternas consecutivas caen en 15 planos del cubo
unitario. Sirve para demostrar que las pruebas de la capa 1 tienen poder real:
si un banco de pruebas aprueba a RANDU, el banco de pruebas no sirve.
"""
from abc import ABC, abstractmethod

import numpy as np


class UniformGenerator(ABC):
    @abstractmethod
    def uniform(self, n: int) -> np.ndarray:
        pass


# (a, c, m, descripcion)
PRESETS = {
    "minstd":  (48271, 0, 2**31 - 1,
                "Park-Miller / MINSTD: terna estandar para modulo primo 2^31-1"),
    "glibc":   (1103515245, 12345, 2**31,
                "glibc: los parametros originales, ahora con SU modulo (2^31)"),
    "randu":   (65539, 0, 2**31,
                "RANDU (IBM): control defectuoso, sus ternas caen en 15 planos"),
}


class LCG(UniformGenerator):
    """
    Generador congruencial lineal, vectorizado por saltos de bloque.

    La recurrencia es secuencial, asi que no se puede vectorizar directamente;
    pero si se aplica B veces se obtiene otra recurrencia lineal

        x_{n+B} = A * x_n + C  (mod m),   A = a^B mod m,
                                          C = c * (a^{B-1} + ... + a + 1) mod m

    que si permite avanzar un bloque entero de B estados de un solo golpe con
    Numpy. Se generan los primeros B estados en un bucle de Python y a partir de
    ahi cada bloque nuevo sale de una sola multiplicacion vectorizada. Con
    B = 8192 eso convierte un millon de iteraciones de Python en 122 operaciones
    de Numpy.

    Los productos caben en int64: el peor caso es (m-1)^2 < 4.7e18, por debajo
    del tope 9.22e18 de int64.
    """

    def __init__(self, seed: int = 12345, a: int = None, c: int = None,
                 m: int = None, bloque: int = 8192, preset: str = "minstd"):
        if a is None or c is None or m is None:
            a_p, c_p, m_p, _ = PRESETS[preset]
            a = a_p if a is None else a
            c = c_p if c is None else c
            m = m_p if m is None else m

        self.a, self.c, self.m = int(a), int(c), int(m)
        self.bloque = int(bloque)
        self.preset = preset

        # Con c = 0 el estado 0 es absorbente (se queda pegado en cero para
        # siempre), asi que la semilla tiene que ser no nula.
        estado = int(seed) % self.m
        if estado == 0 and self.c == 0:
            estado = 1
        self.state = estado

        # Constantes del salto de bloque, calculadas una sola vez con enteros
        # de precision arbitraria de Python.
        self._A = pow(self.a, self.bloque, self.m)
        self._C = (self.c * self._suma_geometrica(self.bloque)) % self.m

    def _suma_geometrica(self, k: int) -> int:
        """
        S(k) = (a^{k-1} + ... + a + 1) mod m, en O(log k) por duplicacion:

            S(2j) = S(j) * (1 + a^j)
            S(2j+1) = 1 + a * S(2j)

        Sumarla termino a termino seria O(k), y con bloques grandes eso hace
        que construir el generador cueste mas que usarlo.
        """
        if k <= 0:
            return 0
        if k == 1:
            return 1 % self.m
        if k % 2 == 0:
            mitad = self._suma_geometrica(k // 2)
            return (mitad * (1 + pow(self.a, k // 2, self.m))) % self.m
        return (1 + self.a * self._suma_geometrica(k - 1)) % self.m

    def _siguientes_escalar(self, n: int) -> list:
        """Bucle directo de la recurrencia; se usa para el primer bloque."""
        estados = []
        estado = self.state
        a, c, m = self.a, self.c, self.m
        for _ in range(n):
            estado = (a * estado + c) % m
            estados.append(estado)
        self.state = estado
        return estados

    def enteros(self, n: int) -> np.ndarray:
        """Los n siguientes estados crudos del generador, en [0, m)."""
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if n <= self.bloque:
            return np.array(self._siguientes_escalar(n), dtype=np.int64)

        B = self.bloque
        bloques = [np.array(self._siguientes_escalar(B), dtype=np.int64)]
        generados = B
        A, C, m = np.int64(self._A), np.int64(self._C), np.int64(self.m)

        while generados < n:
            bloques.append((A * bloques[-1] + C) % m)
            generados += B

        estados = np.concatenate(bloques)[:n]
        self.state = int(estados[-1])
        return estados

    def uniform(self, n: int) -> np.ndarray:
        """
        n uniformes en [0, 1). Se divide entre m (no entre m-1) para que el
        soporte sea [0, 1) y nunca se devuelva exactamente 1.0, que romperia
        cualquier -log(1 - u) o transformada inversa aguas arriba.
        """
        return self.enteros(n).astype(np.float64) / self.m


class MersenneTwisterGenerator(UniformGenerator):
    """Referencia de calidad: el MT19937 de Numpy, con periodo 2^19937 - 1."""

    def __init__(self, seed: int = 12345):
        self.rng = np.random.default_rng(seed)

    def uniform(self, n: int) -> np.ndarray:
        return self.rng.random(n)


def crear_generador_base(nombre: str, semilla: int = 12345) -> UniformGenerator:
    """
    Fabrica el generador uniforme base por nombre.

    Permite que los experimentos cambien de motor aleatorio sin tocar el resto
    del codigo: "mt" para el Mersenne Twister, o cualquier clave de PRESETS
    para el LCG propio.
    """
    if nombre in ("mt", "mersenne", "mt19937"):
        return MersenneTwisterGenerator(semilla)
    if nombre in PRESETS:
        return LCG(semilla, preset=nombre)
    raise ValueError(f"Generador base desconocido: {nombre!r}. "
                     f"Opciones: 'mt', {', '.join(repr(k) for k in PRESETS)}")
