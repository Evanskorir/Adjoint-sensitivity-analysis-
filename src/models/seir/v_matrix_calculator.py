import torch
from src.static.v_matrix_calculator_base import VMatrixCalculatorBase


class VMatrixCalculator(VMatrixCalculatorBase):
    def __init__(self, param: dict, n_age: int, states) -> None:
        super().__init__(param=param, n_age=n_age, states=states)

        self._get_v()

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # e -> e
        v[idx("e"), idx("e")] = self.parameters["gamma"]
        # e -> i
        v[idx("i"), idx("e")] = -self.parameters["gamma"]
        # i -> i
        v[idx("i"), idx("i")] = self.parameters["rho"]

        self.v_inv = torch.linalg.inv(v)

