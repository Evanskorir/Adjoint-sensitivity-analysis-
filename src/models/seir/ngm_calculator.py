import torch
from src.models.seir.v_matrix_calculator import VMatrixCalculator
from src.static.ngm_calculator_base import NGMCalculatorBase


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["e", "i"]
        self.parameters = param
        self.n_states = len(states)

        super().__init__(param=param, n_age=n_age, states=states)
        self.symmetric_contact_matrix = None

        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_state = self.n_states

        f = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["e"]:s_mtx:n_state, i["e"]:s_mtx:n_state] = contact_mtx.T * susc_vec
        f[i["i"]:s_mtx:n_state, i["i"]:s_mtx:n_state] = contact_mtx.T * susc_vec

        return f
