import torch
from src.static.ngm_calculator_base import NGMCalculatorBase
from src.models.british_columbia.v_matrix_calculator import VMatrixCalculator


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["e1", "e2", "i1", "i2"]
        self.parameters = param
        self.n_states = len(states)

        super().__init__(param=param, n_age=n_age, states=states)
        self.symmetric_contact_matrix = None

        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["e1"]:s_mtx:n_states, i["e2"]:s_mtx:n_states] = contact_mtx.T * susc_vec
        f[i["e1"]:s_mtx:n_states, i["i1"]:s_mtx:n_states] = contact_mtx.T * susc_vec
        f[i["e1"]:s_mtx:n_states, i["i2"]:s_mtx:n_states] = contact_mtx.T * susc_vec

        return f
