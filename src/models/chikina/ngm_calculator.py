import torch
from src.models.chikina.v_matrix_calculator import VMatrixCalculator
from src.static.ngm_calculator_base import NGMCalculatorBase


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["i"]
        self.parameters = param
        self.n_states = len(states)

        super().__init__(param=param, n_age=n_age, states=states)
        self.symmetric_contact_matrix = None

        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx

        f = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["i"]:s_mtx:self.n_states, i["i"]:s_mtx:self.n_states] = contact_mtx.T * susc_vec

        return f
