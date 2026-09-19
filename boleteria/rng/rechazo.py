import numpy as np
from .base import NormalGenerator

class RechazoGenerator(NormalGenerator):
    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        # Envolvente exponencial sobre semi-normal (Método A)
        results = np.zeros(n)
        count = 0
        while count < n:
            u1, u2, u3 = self.base_rng.uniform(3)
            x = -np.log(u1) # Exp(1)
            
            if u2 <= np.exp(-(x - 1)**2 / 2):
                z = x if u3 < 0.5 else -x
                results[count] = z
                count += 1
                
        return mu + sigma * results

    def normal_truncada(self, n: int, mu: float, sigma: float, a: float, b: float = np.inf) -> np.ndarray:
        # Envolvente trasladada (Rechazo Adaptado) para z > a_std
        a_std = (a - mu) / sigma
        b_std = (b - mu) / sigma
        
        # Tasa óptima de Robert (1995)
        alpha = (a_std + np.sqrt(a_std**2 + 4)) / 2 if a_std > 0 else 1.0
        
        results = np.zeros(n)
        count = 0
        
        while count < n:
            u1, u2 = self.base_rng.uniform(2)
            # x ~ Exp(alpha) trasladada en a_std
            x = a_std - np.log(u1) / alpha
            
            # Condición de rechazo
            if x <= b_std: 
                rho = np.exp(-0.5 * (x - alpha)**2)
                if u2 <= rho:
                    results[count] = x
                    count += 1
                    
        return mu + sigma * results