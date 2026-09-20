import numpy as np
from .base import NormalGenerator

class RechazoGenerator(NormalGenerator):
    """
    Aceptación-rechazo con envolvente exponencial sobre la semi-normal (Método A).
    La fracción de aceptación teórica es sqrt(2/(pi*e)) ~= 0.76.
    """

    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        """
        Versión vectorizada por lotes: genera bloques de candidatos Exp(1), evalúa la
        condición de aceptación sobre todo el arreglo y les asigna signo con un tercer
        uniforme. Se repite hasta reunir n muestras.
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        resultados = np.empty(n, dtype=np.float64)
        llenos = 0
        while llenos < n:
            faltan = n - llenos
            # Sobredimensionamos por la fracción de aceptación ~0.76.
            m = int(np.ceil(faltan / 0.76 * 1.25)) + 16

            u = self.base_rng.uniform(3 * m).reshape(m, 3)
            x = -np.log(u[:, 0])  # Exp(1)

            aceptados = u[:, 1] <= np.exp(-0.5 * (x - 1.0) ** 2)
            x = x[aceptados]
            signos = np.where(u[aceptados, 2] < 0.5, 1.0, -1.0)
            z = x * signos

            toma = min(z.size, faltan)
            resultados[llenos:llenos + toma] = z[:toma]
            llenos += toma

        return mu + sigma * resultados

    def normal_escalar(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        """Versión de referencia (bucle muestra a muestra). Se conserva para el benchmark."""
        results = np.zeros(n)
        count = 0
        while count < n:
            u1, u2, u3 = self.base_rng.uniform(3)
            x = -np.log(u1)  # Exp(1)

            if u2 <= np.exp(-(x - 1)**2 / 2):
                z = x if u3 < 0.5 else -x
                results[count] = z
                count += 1

        return mu + sigma * results

    def normal_truncada(self, n: int, mu: float, sigma: float, a: float, b: float = np.inf) -> np.ndarray:
        """
        Envolvente exponencial trasladada (rechazo adaptado de Robert, 1995) para la
        cola z > a_std, vectorizado por lotes.
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        a_std = (a - mu) / sigma
        b_std = (b - mu) / sigma

        # Tasa óptima de Robert (1995) para la exponencial trasladada.
        alpha = (a_std + np.sqrt(a_std**2 + 4)) / 2 if a_std > 0 else 1.0

        resultados = np.empty(n, dtype=np.float64)
        llenos = 0
        while llenos < n:
            faltan = n - llenos
            m = faltan + 32

            u = self.base_rng.uniform(2 * m).reshape(m, 2)
            x = a_std - np.log(u[:, 0]) / alpha  # Exp(alpha) trasladada en a_std

            rho = np.exp(-0.5 * (x - alpha) ** 2)
            aceptados = (x <= b_std) & (u[:, 1] <= rho)
            z = x[aceptados]

            toma = min(z.size, faltan)
            resultados[llenos:llenos + toma] = z[:toma]
            llenos += toma

        return mu + sigma * resultados
