from abc import ABC, abstractmethod
import torch


class VMatrixCalculatorBase(ABC):
    def __init__(self, param: dict, n_age: int, states: list) -> None:
        self.states = states
        self.n_states = len(states)
        self.n_age = n_age
        self.parameters = param

        self.i = {self.states[index]: index for index in range(0, self.n_states)}

    def _idx(self, state: str) -> torch.Tensor:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    @abstractmethod
    def _get_v(self):
        pass
