from abc import ABC, abstractmethod
import torch
from src.static.e_matrix_calculator import EMatrixCalculator


class NGMCalculatorBase(ABC):
    def __init__(self, param: dict, n_age: int, states: list) -> None:

        self.ngm_small_tensor = None

        self.states = states
        self.n_states = len(self.states)
        self.n_age = n_age
        self.parameters = param

        self.i = {self.states[index]: index for index in range(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.e_matrix = EMatrixCalculator(n_states=self.n_states, n_age=n_age)
        self.v_matrix = None

    @abstractmethod
    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        pass

    def run(self, symmetric_contact_mtx: torch.Tensor):
        f = self._get_f(contact_mtx=symmetric_contact_mtx)
        ngm_large = torch.matmul(f, self.v_matrix.v_inv)
        self.ngm_small_tensor = torch.matmul(
            torch.matmul(self.e_matrix.e, ngm_large),
            self.e_matrix.e.T
        )
