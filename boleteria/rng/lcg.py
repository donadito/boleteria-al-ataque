import numpy as np
from abc import ABC, abstractmethod

class UniformGenerator(ABC):
    @abstractmethod
    def uniform(self, n: int) -> np.ndarray:
        pass

class LCG(UniformGenerator):
    def __init__(self, seed: int = 12345):
        # Parámetros tipo glibc
        self.m = 2**31 - 1
        self.a = 1103515245
        self.c = 12345
        self.state = seed

    def uniform(self, n: int) -> np.ndarray:
        # Vectorizado
        states = np.zeros(n, dtype=np.float64)
        for i in range(n):
            self.state = (self.a * self.state + self.c) % self.m
            states[i] = self.state / self.m
        return states

class MersenneTwisterGenerator(UniformGenerator):
    def __init__(self, seed: int = 12345):
        self.rng = np.random.default_rng(seed)
        
    def uniform(self, n: int) -> np.ndarray:
        return self.rng.random(n)