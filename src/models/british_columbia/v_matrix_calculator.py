import torch


class VMatrixCalculator:
    def __init__(self, param: dict, n_states: int, n_age: int, states) -> None:
        self.n_states = n_states
        self.states = states
        self.n_age = n_age
        self.parameters = param

        self.i = {self.states[index]: index for index in range(0, self.n_states)}

        self._get_v()

    def _idx(self, state: str) -> torch.Tensor:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E1 -> E1 (exposed to exposed)
        v[idx("e1"), idx("e1")] = self.parameters["H_1"]
        # E1 -> E2 (exposed to asymptomatic)
        v[idx("e2"), idx("e1")] = -self.parameters["H_1"]
        # E2 -> E2
        v[idx("e2"), idx("e2")] = self.parameters["H_2"]
        # E2 -> I1
        v[idx("i1"), idx("e2")] = -self.parameters["H_2"]
        # I1 -> I1
        v[idx("i1"), idx("i1")] = 2 * self.parameters["gamma"]
        # I1 -> I2
        v[idx("i2"), idx("i1")] = - 2 * self.parameters["gamma"]
        # I2 -> I2
        v[idx("i2"), idx("i2")] = 2 * self.parameters["gamma"]

        self.v_inv = torch.linalg.inv(v)

