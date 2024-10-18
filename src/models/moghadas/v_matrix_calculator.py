import torch
from src.static.v_matrix_calculator_base import VMatrixCalculatorBase


class VMatrixCalculator(VMatrixCalculatorBase):
    def __init__(self, param: dict, n_age: int, states) -> None:
        super().__init__(param=param, n_age=n_age, states=states)

        self._get_v()

    def _get_v(self):
        idx = self._idx
        ps = self.parameters
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # e -> e
        v[idx("e"), idx("e")] = ps["sigma"]
        # e -> i_n
        v[idx("i_n"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * (1 - ps["h"]) * ps["sigma"]
        # i_n -> i_n
        v[idx("i_n"), idx("i_n")] = (1 - ps["f_i"]) * ps["gamma"] - ps["f_i"] * ps["tau_i"]
        # e -> q_n
        v[idx("q_n"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * (1 - ps["h"]) * ps["sigma"]
        # i_n -> q_n
        v[idx("q_n"), idx("i_n")] = -ps["f_i"] * ps["tau_i"]
        # q_n -> q_n
        v[idx("q_n"), idx("q_n")] = ps["gamma"]
        # e -> i_h
        v[idx("i_h"), idx("e")] = -(1 - ps["theta"]) * (1 - ps["q"]) * ps["h"] * ps["sigma"]
        # i_h -> i_h
        v[idx("i_h"), idx("i_h")] = (1 - ps["f_i"]) * ps["delta"] - ps["f_i"] * ps["tau_i"]
        # e -> q_h
        v[idx("q_h"), idx("e")] = -(1 - ps["theta"]) * ps["q"] * ps["h"] * ps["sigma"]
        # q_h -> q_h
        v[idx("q_h"), idx("q_h")] = ps["delta"]
        # i_h -> q_h
        v[idx("q_h"), idx("i_h")] = -ps["f_i"] * ps["tau_i"]
        # e -> a_n
        v[idx("a_n"), idx("e")] = -ps["theta"] * ps["sigma"]
        # a_n -> a_n
        v[idx("a_n"), idx("a_n")] = (1 - ps["f_a"]) * ps["gamma"] - ps["f_a"] * ps["tau_a"]
        # a_n -> a_q
        v[idx("a_q"), idx("a_n")] = -ps["f_a"] * ps["tau_a"]
        # a_q -> a_q
        v[idx("a_q"), idx("a_q")] = ps["gamma"]

        self.v_inv = torch.linalg.inv(v)

