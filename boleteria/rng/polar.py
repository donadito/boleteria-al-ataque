import numpy as np
from .base import NormalGenerator

class PolarGenerator(NormalGenerator):
    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        results = np.zeros(n)
        count = 0
        while count < n:
            # Generar pares hasta llenar el arreglo
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
        # Truncamiento por descarte ingenuo (explota en las colas como dice §7)
        results = np.zeros(n)
        count = 0
        while count < n:
            # Pide en bloques pequeños para no gastar de más
            z = self.normal(1, mu, sigma)[0]
            if a <= z <= b:
                results[count] = z
                count += 1
        return results