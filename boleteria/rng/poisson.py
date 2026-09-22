import numpy as np

class LlegadasNHPP:
    def __init__(self, rng_uniforme):
        self.rng = rng_uniforme # Usa el generador base de rng/lcg.py
        self.tau = 25.0
        self.piso = 0.02

    def intensidad_diurna(self, t_minutos: float) -> float:
        # Mezcla de gaussianas para el tráfico de fondo
        t_dia = t_minutos % 1440
        mu1, sigma1, w1 = 750, 55, 1.35   # 12:30
        mu2, sigma2, w2 = 1200, 70, 1.65  # 20:00

        joroba1 = w1 * np.exp(-0.5 * ((t_dia - mu1) / sigma1)**2)
        joroba2 = w2 * np.exp(-0.5 * ((t_dia - mu2) / sigma2)**2)
        return joroba1 + joroba2

    def intensidad(self, t_minutos: float, t_apertura: float, pool: int) -> float:
        # Pico de apertura que decae con tau, sobre el fondo diurno
        amplitud = 0.45 * pool / self.tau
        pico = amplitud * np.exp(-(t_minutos - t_apertura) / self.tau) if t_minutos >= t_apertura else 0.0
        return max(pico + self.intensidad_diurna(t_minutos), self.piso)

    def generar_tramo(self, a: float, b: float, lambda_max: float, intensidad=None) -> list:
        # Adelgazamiento con mayorante por tramos
        f = intensidad if intensidad is not None else self.intensidad_diurna
        llegadas = []
        t = a
        while True:
            u1, u2 = self.rng.uniform(2)
            t += -np.log(u1) / lambda_max # Salto exponencial

            if t >= b:
                break

            # Evaluar intensidad real vs mayorante
            if u2 <= f(t) / lambda_max:
                llegadas.append(t)

        return llegadas

    def generar_fase(self, t_apertura: float, duracion: float, pool: int) -> list:
        """
        Llegadas humanas de una fase completa. La partición es fina durante el pico
        (donde lambda cae de miles a decenas en dos horas) y gruesa en la cola, porque
        una sola cota global rechazaría casi todos los candidatos del período tranquilo.
        """
        # Los cortes internos son fijos (cada 10 min durante las dos horas del
        # pico, luego uno en 480), pero hay que recortarlos a la duracion real y
        # ordenarlos: si la fase dura menos de 480 min, dejarlos tal cual
        # produce un tramo con b < a -- que se salta en silencio -- y otro que
        # se pasa del cierre, generando llegadas fuera de la ventana pedida.
        internos = [d for d in list(np.arange(0.0, 121.0, 10.0)) + [480.0] if 0.0 <= d < duracion]
        cortes = sorted(set(t_apertura + d for d in internos)) + [t_apertura + duracion]
        intensidad = lambda t: self.intensidad(t, t_apertura, pool)

        llegadas = []
        for a, b in zip(cortes[:-1], cortes[1:]):
            lambda_max = 1.05 * max(intensidad(t) for t in np.linspace(a, b, 50))
            llegadas.extend(self.generar_tramo(a, b, lambda_max, intensidad))
        return llegadas
