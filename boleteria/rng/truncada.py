"""
Tercer metodo de generacion: TRANSFORMADA INVERSA para la normal truncada.

Si X ~ N(mu, sigma) restringida a [a, b], su funcion de distribucion es

    F(x) = (Phi(z) - Phi(z_a)) / (Phi(z_b) - Phi(z_a)),   z = (x - mu)/sigma

asi que basta con invertirla sobre un uniforme:

    X = mu + sigma * Phi^{-1}( Phi(z_a) + U * (Phi(z_b) - Phi(z_a)) )

Esto NO rechaza nada: gasta exactamente UN uniforme por muestra, sin importar
donde este la cota. Esa es toda su gracia frente a los dos metodos de rechazo.

El detalle numerico que si importa. Escrita con Phi, la formula se rompe en la
cola derecha: para z_a = 5, Phi(z_a) = 0.99999971..., y al restar dos numeros
tan pegados a 1 se pierden casi todas las cifras significativas (cancelacion
catastrofica). La forma estable es trabajar con la funcion de SUPERVIVENCIA
S(z) = 1 - Phi(z) = P(Z > z), que en la cola derecha vale casi cero y conserva
precision relativa:

    S(X) = S(z_b) + U * (S(z_a) - S(z_b)),   X = mu + sigma * S^{-1}(S(X))

Con b = infinito eso se reduce a X = mu + sigma * S^{-1}(U * S(z_a)), que es
exacto aun para cotas de 8 o 10 sigma. Es la version implementada aqui.

El precio de la inversa es que depende de Phi^{-1}, que no es elemental: se
apoya en la implementacion de scipy (ndtri). Los metodos de rechazo solo
necesitan exp y log. Ese es el intercambio que compara el experimento 1.
"""
import numpy as np
from scipy import special

from .base import NormalGenerator

# Cota inferior para no pasarle 0 exacto a ndtri (que devolveria -infinito).
# Un LCG con c = 0 puede devolver 0.0, asi que la guarda no es decorativa.
_MINIMO = np.finfo(np.float64).tiny


class InversaGenerator(NormalGenerator):
    """
    Normales por transformada inversa, con la variante estable en la cola para
    el caso truncado. Un uniforme por muestra, sin rechazo.
    """

    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        u = np.clip(self.base_rng.uniform(n), _MINIMO, 1.0 - np.finfo(np.float64).epsneg)
        return mu + sigma * special.ndtri(u)

    def normal_truncada(self, n: int, mu: float, sigma: float,
                        a: float, b: float = np.inf) -> np.ndarray:
        """
        Inversa sobre la funcion de supervivencia, que es la parametrizacion que
        no pierde precision cuando la cota se va a la cola.
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        a_std = (a - mu) / sigma
        b_std = (b - mu) / sigma

        s_a = special.ndtr(-a_std)                      # S(a) = P(Z > a)
        s_b = special.ndtr(-b_std) if np.isfinite(b_std) else 0.0

        if s_a - s_b <= 0.0:
            raise ValueError(
                f"Intervalo de truncamiento con probabilidad nula o negativa: "
                f"a={a}, b={b} (S(a)={s_a:.3e}, S(b)={s_b:.3e})")

        u = self.base_rng.uniform(n)
        s = s_b + u * (s_a - s_b)
        s = np.clip(s, _MINIMO, 1.0 - np.finfo(np.float64).epsneg)

        # S^{-1}(s) = -Phi^{-1}(s), porque S(z) = Phi(-z).
        return mu + sigma * (-special.ndtri(s))


def probabilidad_intervalo(mu: float, sigma: float, a: float, b: float = np.inf) -> float:
    """
    P(a <= X <= b) para X ~ N(mu, sigma): la fraccion de aceptacion que tendria
    el descarte ingenuo sobre ese intervalo. Es el numero que hace inviable a
    Polar en la cola, y el denominador del costo teorico del experimento 1.
    """
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    s_a = special.ndtr(-a_std)
    s_b = special.ndtr(-b_std) if np.isfinite(b_std) else 0.0
    return float(s_a - s_b)


def alpha_optimo_robert(a_std: float) -> float:
    """
    Tasa optima de la envolvente exponencial trasladada de Robert (1995) para
    muestrear la cola z > a_std:  alpha* = (a + sqrt(a^2 + 4)) / 2.

    Es la tasa que maximiza la fraccion de aceptacion, y es la que usa
    RechazoGenerator.normal_truncada. Se expone aqui para que el informe pueda
    tabular alpha* y la aceptacion esperada sin duplicar la formula.
    """
    return float((a_std + np.sqrt(a_std ** 2 + 4)) / 2) if a_std > 0 else 1.0
