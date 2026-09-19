from abc import ABC, abstractmethod
import numpy as np

class NormalGenerator(ABC):
    def __init__(self, base_rng):
        self.base_rng = base_rng

    @abstractmethod
    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        pass

    @abstractmethod
    def normal_truncada(self, n: int, mu: float, sigma: float, a: float, b: float) -> np.ndarray:
        pass