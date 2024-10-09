import torch

from src.static.e_matrix_calculator import EMatrixCalculator
from src.models.rost.v_matrix_calculator import VMatrixCalculator


class NGMCalculator:
    def __init__(self, param: dict, n_age: int) -> None:
        self.ngm_small_tensor = None
        self.symmetric_contact_matrix = None
        states = ["l1", "l2", "ip", "a1", "a2", "a3", "i1", "i2", "i3"]
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {self.states[index]: index for index in range(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.n_l = 2
        self.n_a = 3
        self.n_i = 3

        self.e_matrix = EMatrixCalculator(n_states=self.n_states, n_age=n_age)
        self.v_matrix = VMatrixCalculator(param=param, n_states=self.n_states,
                                          n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))
        inf_a = self.parameters["inf_a"] if "inf_a" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0
        inf_p = self.parameters["inf_p"] if "inf_p" in self.parameters.keys() else 1.0

        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["l1"]:s_mtx:n_states, i["ip"]:s_mtx:n_states] = inf_p * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a1"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a2"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a3"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i1"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i2"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i3"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f

    def run(self, symmetric_contact_mtx: torch.Tensor):
        # Calculate large domain NGM
        f = self._get_f(contact_mtx=symmetric_contact_mtx)
        ngm_large = torch.matmul(f, self.v_matrix.v_inv)

        # Calculate small domain NGM
        self.ngm_small_tensor = torch.matmul(
            torch.matmul(self.e_matrix.e, ngm_large),
            self.e_matrix.e.T
        )
