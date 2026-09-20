import numpy as np
from .base import NormalGenerator

class PolarGenerator(NormalGenerator):
    """
    Método polar de Marsaglia. Cada par aceptado (0 < s < 1) produce DOS normales,
    así que la fracción de aceptación teórica es pi/4 ~= 0.785 sobre los pares.
    """

    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        """
        Versión vectorizada por lotes: en vez de pedir uniformes de dos en dos dentro
        de un while, se genera un bloque grande, se aplica la transformada sobre todo
        el arreglo con Numpy y se conservan los aceptados. Se repite hasta llenar n.
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        resultados = np.empty(n, dtype=np.float64)
        llenos = 0
        # Cada par aceptado da 2 normales; sobredimensionamos ~30% para amortiguar
        # la variabilidad del rechazo y evitar iteraciones extra.
        while llenos < n:
            faltan = n - llenos
            pares = int(np.ceil(faltan / 2 / 0.785 * 1.3)) + 16

            u = self.base_rng.uniform(2 * pares).reshape(pares, 2)
            v1 = 2.0 * u[:, 0] - 1.0
            v2 = 2.0 * u[:, 1] - 1.0
            s = v1 * v1 + v2 * v2

            aceptados = (s > 0.0) & (s < 1.0)
            v1, v2, s = v1[aceptados], v2[aceptados], s[aceptados]

            f = np.sqrt(-2.0 * np.log(s) / s)
            # Intercalamos las dos normales que produce cada par aceptado.
            z = np.empty(2 * v1.size, dtype=np.float64)
            z[0::2] = v1 * f
            z[1::2] = v2 * f

            toma = min(z.size, faltan)
            resultados[llenos:llenos + toma] = z[:toma]
            llenos += toma

        return mu + sigma * resultados

    def normal_escalar(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        """Versión de referencia (bucle par-a-par). Se conserva para el benchmark."""
        results = np.zeros(n)
        count = 0
        while count < n:
            u1, u2 = self.base_rng.uniform(2)
            v1, v2 = 2 * u1 - 1, 2 * u2 - 1
            s = v1**2 + v2**2

            if 0 < s < 1:
                f = np.sqrt(-2 * np.log(s) / s)
                z1 = v1 * f
                z2 = v2 * f

                results[count] = z1
                count += 1
                if count < n:
                    results[count] = z2
                    count += 1

        return mu + sigma * results

    def normal_truncada(self, n: int, mu: float, sigma: float, a: float, b: float = np.inf) -> np.ndarray:
        """
        Truncamiento por descarte ingenuo, pero vectorizado por lotes: genera normales
        completas y conserva las que caen en [a, b]. Sigue "explotando" en las colas
        (§7) porque su fracción de aceptación es P(a <= X <= b); lo que cambia es que
        el trabajo se hace sobre arreglos y no muestra por muestra.
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        resultados = np.empty(n, dtype=np.float64)
        llenos = 0
        while llenos < n:
            faltan = n - llenos
            # Estimamos la fracción que sobrevive al truncamiento para no quedarnos cortos.
            z_prueba = self.normal(max(faltan, 256), mu, sigma)
            en_rango = z_prueba[(z_prueba >= a) & (z_prueba <= b)]

            toma = min(en_rango.size, faltan)
            resultados[llenos:llenos + toma] = en_rango[:toma]
            llenos += toma

        return resultados
