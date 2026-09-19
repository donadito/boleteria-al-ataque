import numpy as np

class LlegadasNHPP:
    def __init__(self, rng_uniforme):
        self.rng = rng_uniforme # Usa el generador base de rng/lcg.py
        self.tau = 25.0

    def intensidad_diurna(self, t_minutos: float) -> float:
        # Mezcla de gaussianas para el tráfico de fondo
        t_dia = t_minutos % 1440
        mu1, sigma1, w1 = 750, 55, 1.35   # 12:30
        mu2, sigma2, w2 = 1200, 70, 1.65  # 20:00
        
        joroba1 = w1 * np.exp(-0.5 * ((t_dia - mu1) / sigma1)**2)
        joroba2 = w2 * np.exp(-0.5 * ((t_dia - mu2) / sigma2)**2)
        return joroba1 + joroba2

    def generar_tramo(self, a: float, b: float, lambda_max: float) -> list:
        # Adelgazamiento con mayorante por tramos
        llegadas = []
        t = a
        while True:
            u1, u2 = self.rng.uniform(2)
            t += -np.log(u1) / lambda_max # Salto exponencial
            if t >= b:
                break
            
            # Evaluar intensidad real vs mayorante
            if u2 <= self.intensidad_diurna(t) / lambda_max:
                llegadas.append(t)
                
        return llegadas