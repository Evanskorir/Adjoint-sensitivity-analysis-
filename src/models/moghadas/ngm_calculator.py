import torch
from src.models.moghadas.v_matrix_calculator import VMatrixCalculator
from src.static.ngm_calculator_base import NGMCalculatorBase


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["e", "i_n", "q_n", "i_h", "q_h", "a_n", "a_q"]
        self.parameters = param
        self.n_states = len(states)

        super().__init__(param=param, n_age=n_age, states=states)
        self.symmetric_contact_matrix = None

        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        inf_a = self.parameters["k"] if "k" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0

        f[i["e"]:s_mtx:n_states, i["a_n"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["a_q"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f
