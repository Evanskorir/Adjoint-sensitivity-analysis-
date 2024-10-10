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
        v[idx("i"), idx("i")] = self.parameters["alpha_i"]
        self.v_inv = torch.linalg.inv(v)

